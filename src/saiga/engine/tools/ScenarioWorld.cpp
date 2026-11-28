/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ScenarioWorld: the world of a scenario with its helpers, the step observation and the save/load swap.

#include "saiga/engine/tools/ScenarioWorld.h"

#include "saiga/engine/content/ContentCache.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/net/NetOwner.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/save/MemorySaveFileSystem.h"
#include "saiga/engine/save/SaveDiff.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/terrain/TerrainEditStore.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace Saiga
{
namespace Engine
{
namespace
{
WorldSetup scenarioSetup(WorldSetup s)
{
    // No clock calls: a scenario is reproducible (R-DET-1)
    s.profiling = false;
    // Content within 128 m of a player is stamped (§17.2; S23a): a scenario that needs content farther away moves its
    // Player there
    for (SpaceTemplate& t : s.config.spaceTemplates) t.fullRadius = std::min(t.fullRadius, 128.0);
    // S24 (§17.2): the simulation radius at most 128 m (and the lowered fullRadius), a residencyRadius clamped to it,
    // no prefetch: a scenario world's required and requested distance is fullRadius
    for (SpaceTemplate& t : s.config.spaceTemplates)
    {
        t.simulationRadius = std::min({t.simulationRadius, 128.0, t.fullRadius});
        if (t.residencyRadius != 0) t.residencyRadius = std::min(t.residencyRadius, t.fullRadius);
        t.prefetchMargin = 0;
    }
    // The worlds of a scenario (host, clients, the save/load replacement) share one content cache (§17.2; one per
    // setup: the runner passes its own to every run)
    if (!s.sharedContent) s.sharedContent = std::make_shared<ContentCache>();
    return s;
}
}  // namespace

ScenarioWorld::ScenarioWorld(const WorldSetup& s, const ScenarioOptions& o, const ScenarioNetwork& n)
    : worldSetup(scenarioSetup(s)), options(o), network(n), asyncRng(hashCombine(s.seed, 0x5CE7A110ull))
{
    logStart = engineLog().lastSequence();
    if (network.enabled)
    {
        // The host (peer 0) and the clients over a loopback network without loss (§17.3); reproducible (R-DET-1)
        LoopbackOptions lo;
        lo.delayRounds = network.delaySteps;
        lo.seed        = hashCombine(worldSetup.seed, 0x4E7ull);
        transports     = LoopbackTransport::createNetwork(1 + std::max(1, options.clients), lo);
        WorldSetup hs  = worldSetup;
        hs.transport   = transports[0].get();
        hs.role        = NetRole::Host;
        hs.localPeer   = 0;
        w              = std::make_unique<World>(hs);
    }
    else
        w = std::make_unique<World>(worldSetup);
    // A replacement world shares the data (also after reload_data)
    worldSetup.sharedData = w->sharedDataStore();
    w->setTickEveryStep(true);
    // S24 (§8.6): synchronous chunk loading in every world of a scenario (reproducible, R-DET-1)
    w->setStreamingSynchronous(true);
    // Every world of the scenario has its own recording sink, so the extract runs in every scenario (§17.2, S19b)
    hostSink = giveSink(*w);
    if (network.enabled)
    {
        auto report = [this](const std::string& m)
        {
            if (failed.size() < 200) failed.push_back("step " + std::to_string(observedSteps) + ": " + m);
        };
        w->replication().setWriteDetector(report);
        for (size_t i = 1; i < transports.size(); ++i)
        {
            WorldSetup cs = worldSetup;
            cs.transport  = transports[i].get();
            cs.role       = NetRole::Client;
            cs.localPeer  = uint8_t(i);
            clientWorlds.push_back(std::make_unique<World>(cs));
            clientWorlds.back()->setTickEveryStep(true);
            clientWorlds.back()->setStreamingSynchronous(true);
            clientSinks.push_back(giveSink(*clientWorlds.back()));
            clientWorlds.back()->replication().setWriteDetector(report);
            clientSteps.push_back(0);
        }
        joinClients();
    }
    attach();
}

void ScenarioWorld::joinClients()
{
    // S26 (§17.3): network-only steps (no host step: its step count and hash are unchanged) until every client joined
    // (the handshake checks the render tables, the schema and the generation) and, in a world without streaming, its
    // `all` subscriptions are indexed and the initial snapshots applied (a client-owned character's first steps stand
    // on the edited surface)
    auto joined = [&]()
    {
        for (auto& c : clientWorlds)
            if (c->joinState() != JoinState::Joined) return false;
        return true;
    };
    auto rejected = [&]()
    {
        for (auto& c : clientWorlds)
            if (c->joinState() == JoinState::Rejected) return true;
        return false;
    };
    const int bound = 4 * network.delaySteps + 8;
    int pumps       = 0;
    for (; pumps < bound && !joined() && !rejected(); ++pumps)
    {
        w->stepNetworkOnly();
        for (auto& c : clientWorlds) c->stepNetworkOnly();
    }
    if (!joined())
    {
        reportJoinFailures("join");
        if (!joinFailureReported)
            failed.push_back("join: the clients did not join within " + std::to_string(bound) + " network steps");
        return;
    }
    for (int k = 0; k < 600 && !chunksSynced(); ++k)
    {
        w->stepNetworkOnly();
        for (auto& c : clientWorlds) c->stepNetworkOnly();
    }
    if (!chunksSynced()) failed.push_back("join: the initial chunk snapshots did not arrive within 600 network steps");
}

void ScenarioWorld::reportJoinFailures(const std::string& when)
{
    for (size_t i = 0; i < clientWorlds.size() && !joinFailureReported; ++i)
        if (clientWorlds[i]->joinState() == JoinState::Rejected)
        {
            joinFailureReported = true;
            failed.push_back(when + ": client " + std::to_string(i + 1) +
                             " was rejected: " + clientWorlds[i]->joinRejection());
        }
}

void ScenarioWorld::holdChunk(int peer, const ChunkId& chunk)
{
    if (network.enabled) client(peer).replication().chunkSync().hold(chunk);
}

void ScenarioWorld::releaseChunk(int peer, const ChunkId& chunk)
{
    if (network.enabled) client(peer).replication().chunkSync().release(chunk);
}

World& ScenarioWorld::client(int peer)
{
    SAIGA_ASSERT(peer >= 1 && size_t(peer) <= clientWorlds.size(),
                 "ScenarioWorld::client: no client " + std::to_string(peer));
    return *clientWorlds[size_t(peer - 1)];
}

Entity ScenarioWorld::onClient(int peer, Entity hostEntity)
{
    const NetId id = w->replication().netIdOf(hostEntity);
    return id == 0 ? Entity{} : client(peer).replication().entityOfNetId(id);
}

void ScenarioWorld::ownPlayer(Entity e, uint8_t peer)
{
    if (network.enabled && peer != 0) w->setOwner(e, peer);
}

void ScenarioWorld::edit(Entity entity, std::function<void(World&, Entity)> fn)
{
    const uint8_t owner = network.enabled && w->alive(entity) ? w->ownerOf(entity) : 0;
    if (owner == 0)
    {
        fn(*w, entity);
        return;
    }
    if (!check(size_t(owner) <= clientWorlds.size(),
               "edit: the owner peer " + std::to_string(owner) + " is not a client of this run"))
        return;
    pendingEdits.push_back(PendingEdit{entity, owner, std::move(fn)});
    applyPending();
}

void ScenarioWorld::applyPending()
{
    // In order; an edit waits for its entity, and later edits of the same entity wait behind it
    std::vector<Entity> blocked;
    for (size_t i = 0; i < pendingEdits.size();)
    {
        PendingEdit& p = pendingEdits[i];
        const Entity e = onClient(p.peer, p.entity);
        if (!e.valid() || std::find(blocked.begin(), blocked.end(), p.entity) != blocked.end())
        {
            blocked.push_back(p.entity);
            ++i;
            continue;
        }
        auto fn            = std::move(p.fn);
        const uint8_t peer = p.peer;
        pendingEdits.erase(pendingEdits.begin() + std::ptrdiff_t(i));
        fn(client(peer), e);
    }
}

void ScenarioWorld::stepClients()
{
    hostSteps++;
    for (size_t i = 0; i < clientWorlds.size(); ++i)
    {
        int n = 1;
        if (network.async)
        {
            // The client starts 3 steps late; the cumulative difference to the host stays in [-2, 2]
            const int64_t target = int64_t(hostSteps) - 3;
            std::vector<int> allowed;
            for (int k = 0; k <= 2; ++k)
            {
                const int64_t d = int64_t(clientSteps[i]) + k - target;
                if (d >= -2 && d <= 2) allowed.push_back(k);
            }
            n = allowed.empty() ? 1 : allowed[size_t(asyncRng.range(0, int64_t(allowed.size()) - 1))];
        }
        for (int k = 0; k < n; ++k)
        {
            clientWorlds[i]->step();
            clientSteps[i]++;
        }
    }
    applyPending();
}

RecordingRenderSink* ScenarioWorld::giveSink(World& world)
{
    sinks.push_back(std::make_unique<RecordingRenderSink>());
    world.setRenderSink(sinks.back().get());
    // S23b: content render sets around the players on every peer (§17.2)
    ContentRenderSettings content;
    content.centres = ContentCentres::Players;
    world.setContentRenderSettings(content);
    return sinks.back().get();
}

RecordingRenderSink& ScenarioWorld::renderSink(int peer)
{
    SAIGA_ASSERT(peer >= 0 && size_t(peer) <= clientSinks.size(), "ScenarioWorld::renderSink: no such peer");
    return peer == 0 ? *hostSink : *clientSinks[size_t(peer) - 1];
}

ScenarioWorld::~ScenarioWorld()
{
    // The API detaches from the world before the world goes away
    apiPtr.reset();
    // The extract consumers go with the sinks (§17.2)
    for (auto& c : clientWorlds) c->setRenderSink(nullptr);
    if (w) w->setRenderSink(nullptr);
    for (auto& r : retiredWorlds) r->setRenderSink(nullptr);
    clientWorlds.clear();
    if (w && stepListener >= 0) w->removeStepListener(stepListener);
    if (w && countListener >= 0) w->removeStepListener(countListener);
    for (int id : ownListeners)
        if (w) w->removeStepListener(id);
}

void ScenarioWorld::attachCounter()
{
    countListener = w->addStepListener(
        [this]()
        {
            observedSteps++;
            return true;
        });
}

void ScenarioWorld::moveObserverLast()
{
    // Not while the listeners run: a listener added now runs from the next step on, and the observer must not miss
    // this one
    if (w->dispatchingListeners()) return;
    w->removeStepListener(stepListener);
    attachObserver();
}

void ScenarioWorld::attachObserver()
{
    stepListener = w->addStepListener(
        [this]()
        {
            try
            {
                onStep();
            }
            catch (...)
            {
                // The world removes a throwing listener: the observation continues in a new registration (adding
                // listeners while they run is allowed), so later steps are still counted and still end the scenario
                attachObserver();
                throw;
            }
            return true;
        });
}

void ScenarioWorld::attach()
{
    attachCounter();
    ownListeners.clear();
    for (auto& fn : listeners)
        ownListeners.push_back(w->addStepListener(
            [fn]()
            {
                (*fn)();
                return true;
            }));
    attachObserver();
}

void ScenarioWorld::addStepListener(std::function<void()> fn)
{
    auto shared = std::make_shared<std::function<void()>>(std::move(fn));
    listeners.push_back(shared);
    ownListeners.push_back(w->addStepListener(
        [shared]()
        {
            (*shared)();
            return true;
        }));
    moveObserverLast();
}

void ScenarioWorld::onStep()
{
    // After the scenario's listeners (the counter already counted this step). Throws ScenarioAbort past maxSteps (the
    // observer registers itself again, see attachObserver)
    // The clients step with the host, before the hash and the traces (§17.3)
    if (network.enabled)
    {
        stepClients();
        // S23a, S26: a client whose join failed (or whose adoption of a host load's generation failed) fails the
        // variant (once)
        reportJoinFailures("step " + std::to_string(observedSteps));
    }
    if (control.hashes) control.hashes->push_back(w->hash());
    for (Trace& t : traces)
    {
        t.values.push_back({observedSteps, t.read(*w)});
        if (t.values.size() > 20) t.values.pop_front();
    }
    if (observedSteps == control.stopAt)
    {
        if (control.stopSave) *control.stopSave = w->save();
        if (abortReason.empty()) abortReason = "stopped by the runner at step " + std::to_string(observedSteps);
    }
    if (observedSteps > options.maxSteps && abortReason.empty())
        abortReason = "more than " + std::to_string(options.maxSteps) + " steps";
    if (options.deadline && abortReason.empty() && std::chrono::steady_clock::now() >= *options.deadline)
        abortReason = "stopped: tool time budget";
    // Ends the scenario also when it steps outside the helpers (world().step(), advance()): the exception leaves the
    // world's step; the world API turns it into an error result, after which call() ends the scenario
    if (!abortReason.empty()) throw ScenarioAbort{abortReason};
}

void ScenarioWorld::checkAbort() const
{
    if (!abortReason.empty()) throw ScenarioAbort{abortReason};
}

WorldApi& ScenarioWorld::api()
{
    if (!apiPtr)
    {
        apiPtr = std::make_unique<WorldApi>(*w);
        // No wall-clock limit on step and apply (a scenario must not depend on the machine's speed); maxSteps bounds
        // the run
        apiPtr->setTimeBudget(1e7);
    }
    return *apiPtr;
}

ApiResult ScenarioWorld::call(const std::string& operation, const json11::Json& arguments)
{
    checkAbort();
    ApiResult r = api().call(operation, arguments);
    checkAbort();
    // The API's step counts like a step helper for the save/load variant's swap (after the call, between steps)
    if (!control.saveLoadDone && observedSteps >= control.saveLoadAt && stepDepth == 0 && !w->dispatchingListeners())
        saveLoad();
    return r;
}

void ScenarioWorld::involve(Entity e)
{
    if (involved.size() < 10 && std::find(involved.begin(), involved.end(), e) == involved.end()) involved.push_back(e);
}

Entity ScenarioWorld::spawnPlayer(const Vec3& position, std::optional<uint8_t> peer)
{
    const Entity e = w->create();
    w->get<Transform>(e).setPosition(position);
    Player p;
    p.peer        = playerPeer(peer);
    p.characterId = nextCharacterId++;
    w->add<Player>(e, p);
    w->add<PlayerInput>(e);
    w->add<CharacterIntent>(e);
    ownPlayer(e, p.peer);
    involve(e);
    return e;
}

void ScenarioWorld::move(Entity player, const Vec3& direction)
{
    if (!check(w->alive(player) && w->has<PlayerInput>(player), "move: the player is not alive or has no PlayerInput"))
        return;
    const double n   = direction.norm();
    const Vec3 moved = n > 1 ? Vec3(direction / n) : direction;
    edit(player,
         [moved](World& world, Entity e)
         {
             if (world.alive(e) && world.has<PlayerInput>(e)) world.get<PlayerInput>(e).move = moved;
         });
}

void ScenarioWorld::input(Entity player, uint32_t action, const Vec3& aim)
{
    if (!check(w->alive(player) && w->has<Player>(player), "input: the player is not alive")) return;
    edit(player,
         [action, aim](World& world, Entity e)
         {
             if (world.alive(e)) world.queueAction(PlayerAction{e, action, aim});
         });
}

void ScenarioWorld::damage(Entity target, double amount, Entity source)
{
    w->events.emit(DamageEvent{target, source, amount});
}

void ScenarioWorld::enableDistanceTicking()
{
    tickEverything = false;
    w->setTickEveryStep(false);
}

void ScenarioWorld::step(int n)
{
    if (w->dispatchingListeners())
    {
        failed.push_back("step " + std::to_string(observedSteps) +
                         ": a step helper was called from a step listener (not supported: step in the scenario "
                         "function)");
        abortReason = "stepped inside a step listener";
        throw ScenarioAbort{abortReason};
    }
    for (int i = 0; i < n; ++i)
    {
        checkAbort();
        // Not inside a step listener (a listener that steps): the old world must not be replaced while it runs them
        if (!control.saveLoadDone && observedSteps >= control.saveLoadAt && stepDepth == 0) saveLoad();
        stepDepth++;
        try
        {
            w->step();
        }
        catch (...)
        {
            stepDepth--;
            throw;
        }
        stepDepth--;
    }
    checkAbort();
}

void ScenarioWorld::stepSeconds(double seconds)
{
    step(int(std::llround(seconds / w->dt())));
}

bool ScenarioWorld::stepUntil(const std::function<bool()>& predicate, int maxSteps)
{
    for (int i = 0; i < maxSteps; ++i)
    {
        if (predicate()) return true;
        step();
    }
    return predicate();
}

bool ScenarioWorld::check(bool condition, const std::string& message)
{
    if (!condition) failed.push_back("step " + std::to_string(observedSteps) + ": " + message);
    return condition;
}

void ScenarioWorld::editTerrain(const TerrainBrush& brush, SpaceId space)
{
    checkAbort();
    SAIGA_LOG("scenario", "step %llu: editTerrain space %d, op %s, center (%g, %g, %g), radius %g, strength %g",
              (unsigned long long)observedSteps, int(space), TerrainBrush::opNames[brush.op % TerrainBrush::Count],
              brush.center.x(), brush.center.y(), brush.center.z(), brush.radius, brush.strength);
    w->editTerrain(space, brush);
}

double ScenarioWorld::terrainHeight(double x, double z, std::optional<int> peer, SpaceId space)
{
    World& world = peer && network.enabled ? client(*peer) : *w;
    return world.terrainHeight(space, x, z);
}

int32_t ScenarioWorld::terrainSample(int64_t i, int64_t k, std::optional<int> peer, SpaceId space)
{
    World& world = peer && network.enabled ? client(*peer) : *w;
    return world.terrainSample(space, i, k);
}

bool ScenarioWorld::chunksSynced()
{
    if (w->replication().subscriptions().queuedSnapshots() != 0) return false;
    const TerrainEditStore& h = w->terrainEdits();
    for (auto& c : clientWorlds)
    {
        if (c->joinState() != JoinState::Joined) continue;
        const ClientChunkSync& sync = c->replication().chunkSync();
        if (sync.pendingCount() != 0 || sync.queuedParts() != 0) return false;
        if (!c->streaming() && !sync.allIndexed()) return false;
        const TerrainEditStore& s = c->terrainEdits();
        if (c->streaming())
        {
            for (const auto& [chunk, e] : sync.chunks())
                if (e.state == ClientChunkSync::State::Synced && s.sequence(chunk) != h.sequence(chunk)) return false;
            continue;
        }
        // Without streaming every chunk is Synced: the chunks with state on either side
        for (size_t i = 0; i < h.spaceCount(); ++i)
        {
            for (const auto& kv : h.chunks(SpaceId(i)))
                if (s.sequence(kv.first) != kv.second.sequence) return false;
            for (const auto& kv : s.chunks(SpaceId(i)))
                if (h.sequence(kv.first) != kv.second.sequence) return false;
        }
    }
    return true;
}

void ScenarioWorld::compareChunks()
{
    const TerrainEditStore& h = w->terrainEdits();
    int reported              = 0;
    auto fail                 = [&](const std::string& m)
    {
        if (reported++ < 10) check(false, "chunk replicas differ: " + m);
    };
    std::vector<int32_t> da, db;
    for (size_t p = 0; p < clientWorlds.size(); ++p)
    {
        const World& cw             = *clientWorlds[p];
        const std::string peer      = "peer " + std::to_string(p + 1);
        const World::NetStats& ns   = cw.netStats();
        const ClientChunkSync& sync = cw.replication().chunkSync();
        check(ns.modificationGaps == 0, peer + ": " + std::to_string(ns.modificationGaps) + " modification gaps");
        check(ns.chunkRerequests == 0, peer + ": " + std::to_string(ns.chunkRerequests) + " chunk re-requests");
        check(sync.stats().queueOverflows == 0,
              peer + ": " + std::to_string(sync.stats().queueOverflows) + " queue overflows");
        const TerrainEditStore& s = cw.terrainEdits();
        // The chunks to compare: without streaming every chunk with state on either side (all Synced); with streaming
        // the Synced chunks of the client's table
        std::set<ChunkId> chunks;
        if (!cw.streaming())
        {
            for (size_t i = 0; i < h.spaceCount(); ++i)
            {
                for (const auto& kv : h.chunks(SpaceId(i))) chunks.insert(kv.first);
                for (const auto& kv : s.chunks(SpaceId(i))) chunks.insert(kv.first);
            }
            for (const auto& kv : w->contentRecords().records()) chunks.insert(kv.first);
            for (const auto& kv : cw.contentRecords().records()) chunks.insert(kv.first);
        }
        else
        {
            for (const auto& [c, e] : sync.chunks())
                if (e.state == ClientChunkSync::State::Synced) chunks.insert(c);
            // A chunk that is not subscribed has no state on the client (D10)
            for (size_t i = 0; i < s.spaceCount(); ++i)
                for (const auto& kv : s.chunks(SpaceId(i)))
                    if (!sync.chunks().count(kv.first))
                        fail(peer + ": space " + std::to_string(kv.first.space) + " chunk (" +
                             std::to_string(kv.first.x) + ", " + std::to_string(kv.first.z) +
                             ") has terrain state but is not subscribed");
            for (const auto& kv : cw.contentRecords().records())
                if (!sync.chunks().count(kv.first))
                    fail(peer + ": space " + std::to_string(kv.first.space) + " chunk (" + std::to_string(kv.first.x) +
                         ", " + std::to_string(kv.first.z) + ") has a record but is not subscribed");
        }
        for (const ChunkId& c : chunks)
        {
            const std::string what = peer + ", space " + std::to_string(c.space) + " chunk (" + std::to_string(c.x) +
                                     ", " + std::to_string(c.z) + ")";
            const uint32_t sa = h.sequence(c), sb = s.sequence(c);
            if (sa != sb)
            {
                fail(what + ": sequence " + std::to_string(sb) + ", the host's " + std::to_string(sa));
                continue;
            }
            const ChunkRecord ra = w->effectiveRecord(c), rb = cw.effectiveRecord(c);
            if (ra != rb)
                fail(what + ": placement version " + std::to_string(rb.placementVersion) + " with " +
                     std::to_string(rb.tombstones.size()) + " tombstones, the host's " +
                     std::to_string(ra.placementVersion) + " with " + std::to_string(ra.tombstones.size()));
            if (w->spaceTemplate(c.space).kind != SpaceKind::OpenWorld || (!h.hasDeltas(c) && !s.hasDeltas(c)))
                continue;
            h.denseDelta(c, da);
            s.denseDelta(c, db);
            for (size_t k = 0; k < da.size(); ++k)
                if (da[k] != db[k])
                {
                    const size_t N = size_t(h.chunkSamples(c.space) + 1);
                    fail(what + " (sequence " + std::to_string(sa) + "): the sample (" + std::to_string(k % N) + ", " +
                         std::to_string(k / N) + ") has the delta " + std::to_string(db[k]) + ", the host's " +
                         std::to_string(da[k]));
                    break;
                }
        }
    }
}

double ScenarioWorld::distance(Entity a, Entity b)
{
    if (!check(w->alive(a) && w->alive(b), "distance: an entity is not alive"))
        return std::numeric_limits<double>::infinity();
    return (w->read<Transform>(a).position() - w->read<Transform>(b).position()).norm();
}

void ScenarioWorld::saveLoad()
{
    // The old world must not be replaced while it runs its listeners (the rest of them would see the old world)
    if (w->dispatchingListeners() || w->inStep())
    {
        failed.push_back("step " + std::to_string(observedSteps) + ": saveLoad was called from " +
                         (w->inStep() ? "a system or game object during a step" : "a step listener") +
                         " (not supported: call it in the scenario function)");
        abortReason = "save/load inside a step";
        throw ScenarioAbort{abortReason};
    }
    if (network.enabled)
    {
        // The clients are connected to the old host world (a reload would need a new session with them)
        failed.push_back("step " + std::to_string(observedSteps) +
                         ": saveLoad is not supported in the host + client variants");
        abortReason = "save/load in a host + client variant";
        throw ScenarioAbort{abortReason};
    }
    control.saveLoadDone             = true;
    control.swappedAtStep            = observedSteps;
    const std::vector<uint8_t> saved = w->save();
    auto fresh                       = std::make_unique<World>(worldSetup);
    // Before the load: configure() and onLoad() of the loaded game objects see the scenario's tick setting
    fresh->setTickEveryStep(w->tickEveryStep());
    // S24: synchronous chunk loading (the load's residency rebuild too)
    fresh->setStreamingSynchronous(true);
    // Its own sink before the load: the load's structural records create the instances (S19b)
    RecordingRenderSink* freshSink = giveSink(*fresh);
    const LoadResult r             = fresh->tryLoad(saved);
    if (!r.committed)
    {
        failed.push_back("step " + std::to_string(observedSteps) + ": save/load: the load was rejected: " + r.error);
        throw ScenarioAbort{"the save/load round trip failed"};
    }
    for (const LoadResult::HookError& e : r.hookErrors)
        failed.push_back("step " + std::to_string(observedSteps) + ": save/load: the game object of entity " +
                         std::to_string(e.entity.index) + ":" + std::to_string(e.entity.generation) + " (" +
                         e.className + ") failed: " + e.message);
    for (const std::string& e : r.listenerErrors)
        failed.push_back("step " + std::to_string(observedSteps) + ": save/load: a load listener failed: " + e);
    const std::vector<uint8_t> again = fresh->save();
    if (again != saved)
    {
        DiffFilter f;
        f.limit          = 20;
        const SaveDiff d = diffSaves(saved, again, f);
        std::string text;
        for (const SaveDifference& x : d.differences) text += "\n  " + x.describe();
        failed.push_back("step " + std::to_string(observedSteps) +
                         ": save/load: the loaded world saves differently (state outside the saved components, "
                         "R-STATE-1):" +
                         text);
    }
    // S25: the same state through a save directory (a memory file system): the old world adopts it (commit 1,
    // synchronous IO) and closes it, a fresh world of the same setup opens it and must save the blob
    if (options.directoryRoundTrip)
    {
        auto fs = std::make_shared<MemorySaveFileSystem>();
        SaveDirectoryOptions o;
        o.fileSystem                 = fs;
        o.synchronousIo              = true;
        o.commitIntervalSeconds      = 0;
        o.zstdLevel                  = 1;
        const SaveOpenResult created = w->openSaveDirectory("scenario", o);
        if (!created.ok)
            failed.push_back("step " + std::to_string(observedSteps) +
                             ": save/load: the directory round trip could not create the directory: " + created.error);
        else
        {
            w->closeSaveDirectory();
            World other(worldSetup);
            other.setTickEveryStep(w->tickEveryStep());
            other.setStreamingSynchronous(true);
            const SaveOpenResult opened = other.openSaveDirectory("scenario", o);
            if (!opened.ok)
                failed.push_back("step " + std::to_string(observedSteps) +
                                 ": save/load: the directory round trip could not open the directory: " + opened.error);
            else
            {
                const std::vector<uint8_t> viaDirectory = other.save();
                other.closeSaveDirectory();
                if (viaDirectory != saved)
                {
                    DiffFilter f;
                    f.limit          = 20;
                    const SaveDiff d = diffSaves(saved, viaDirectory, f);
                    std::string text;
                    for (const SaveDifference& x : d.differences) text += "\n  " + x.describe();
                    failed.push_back("step " + std::to_string(observedSteps) +
                                     ": save/load: the directory round trip differs:" + text);
                }
            }
        }
    }
    // Step listeners the runner does not know would be lost with the old world: fail instead of losing them silently
    // (by ID: a scenario listener the world removed after it threw must not hide a direct one)
    bool unknown = false;
    for (int id : w->stepListenerIds())
        unknown = unknown || (id != stepListener && id != countListener &&
                              std::find(ownListeners.begin(), ownListeners.end(), id) == ownListeners.end() &&
                              !(apiPtr && id == apiPtr->stepListenerId()));
    if (unknown)
        failed.push_back("step " + std::to_string(observedSteps) +
                         ": save/load: the world has step listeners that are lost with it (add them with "
                         "ScenarioWorld::addStepListener)");
    // What is not world state carries over
    fresh->setPaused(w->isPaused());
    fresh->setTimeScale(w->timeScale());
    fresh->setAccumulatedTime(w->accumulatedTime());
    fresh->setWorkerThreads(w->workerThreads());
    w->removeStepListener(stepListener);
    w->removeStepListener(countListener);
    for (int id : ownListeners) w->removeStepListener(id);
    if (apiPtr) apiPtr->rebind(*fresh);
    std::swap(w, fresh);
    // The replaced world's extract stops (its consumer is removed)
    fresh->setRenderSink(nullptr);
    hostSink = freshSink;
    attach();
    retiredWorlds.push_back(std::move(fresh));  // the old world lives until the scenario ends
}

std::string ScenarioWorld::report() const
{
    // The context of the failures (the failures themselves are listed by the runner)
    std::string out;
    const std::vector<LogRing::Line> lines = engineLog().since(logStart);
    if (!lines.empty())
    {
        out += "  log:\n";
        const size_t first = lines.size() > 200 ? lines.size() - 200 : 0;
        for (size_t i = first; i < lines.size(); ++i)
            out += "    [" + lines[i].category + "] " + lines[i].message + "\n";
    }
    for (Entity e : involved)
        out += "  entity " + std::to_string(e.index) + ":" + std::to_string(e.generation) + ": " +
               (w->alive(e) ? w->dumpJson(e) : std::string("not alive")) + "\n";
    for (const Trace& t : traces)
    {
        out += "  trace " + std::string(t.component) + " of " + std::to_string(t.entity.index) + ":\n";
        for (const auto& v : t.values) out += "    step " + std::to_string(v.first) + ": " + v.second.dump() + "\n";
    }
    return out;
}

// ---- the end of a host + client run (§17.3)
// --------------------------------------------------------------------------

void ScenarioWorld::finishNetwork()
{
    if (!network.enabled) return;
    for (const PendingEdit& p : pendingEdits)
        check(false, "an edit of entity " + std::to_string(p.entity.index) + " never reached its owner (peer " +
                         std::to_string(p.peer) + ")");
    pendingEdits.clear();
    // The owners whose final state the host must receive: the next owner state each one sends
    std::vector<uint64_t> finalOwnerState(clientWorlds.size() + 1, 0);
    std::vector<NetId> known = w->replication().netIds();
    std::sort(known.begin(), known.end());
    for (NetId id : known)
    {
        const uint8_t owner = w->ownerOf(w->replication().entityOfNetId(id));
        if (owner != 0 && size_t(owner) <= clientWorlds.size())
            finalOwnerState[owner] = client(owner).replication().ownerStateSequence(owner) + 1;
    }
    // Client-owned entities published during the sync (created after the scenario's last step): their owner's state
    // counts once the owner has the entity (the state it sends in the step it receives the spawn)
    std::vector<std::pair<uint8_t, NetId>> late;
    // Host and clients 1:1 until the host has the owners' final states and every client applied a snapshot sent after
    // that (§17.3; without loss at most 2 * delaySteps + 3 steps, and delaySteps + 1 more for a late spawn)
    int bound       = 2 * network.delaySteps + 3;
    bool ownersDone = false;
    uint64_t needed = 0;
    int steps       = 0;
    int entitySteps = -1;  // the step the entity and owner-state sync finished
    bool synced     = false;
    while (steps < 600 && !synced)
    {
        steps++;
        w->stepNetworkOnly();
        for (NetId id : w->replication().netIds())
        {
            if (std::binary_search(known.begin(), known.end(), id)) continue;
            known.insert(std::upper_bound(known.begin(), known.end(), id), id);
            const uint8_t owner = w->ownerOf(w->replication().entityOfNetId(id));
            if (owner == 0 || size_t(owner) > clientWorlds.size()) continue;
            if (late.empty()) bound += network.delaySteps + 1;
            late.push_back({owner, id});
            ownersDone = false;
        }
        if (!ownersDone && late.empty())
        {
            ownersDone = true;
            for (size_t p = 1; p < finalOwnerState.size(); ++p)
                if (finalOwnerState[p] != 0 && w->replication().ownerStateSequence(uint8_t(p)) < finalOwnerState[p])
                    ownersDone = false;
            if (ownersDone) needed = w->replication().snapshotSequence();
        }
        for (auto& c : clientWorlds) c->stepNetworkOnly();
        for (size_t i = 0; i < late.size();)
        {
            World& owner = client(late[i].first);
            if (!owner.replication().entityOfNetId(late[i].second).valid())
            {
                ++i;
                continue;
            }
            finalOwnerState[late[i].first] =
                std::max(finalOwnerState[late[i].first], owner.replication().ownerStateSequence(late[i].first));
            late.erase(late.begin() + std::ptrdiff_t(i));
        }
        if (ownersDone)
        {
            bool entities = true;
            for (auto& c : clientWorlds) entities = entities && c->replication().snapshotSequence() >= needed;
            // The entity and owner-state sync has its timing bound; the chunk drain after it (S26: snapshots are
            // budgeted per step) only the 600-step limit
            if (entities && entitySteps < 0) entitySteps = steps;
            synced = entities && chunksSynced();
        }
    }
    // A failure received during the drain fails the variant too (once; PR review of S23a: only onStep reported it)
    reportJoinFailures("final network sync");
    if (!check(synced, "the final network sync did not finish in 600 steps")) return;
    check(entitySteps <= bound, "the final network sync of the entities took " + std::to_string(entitySteps) +
                                    " steps (at most " + std::to_string(bound) + " without loss)");
    compareReplicas();
}

void ScenarioWorld::compareReplicas()
{
    compareChunks();
    // For every replicated component the authority's value (the host, or the owner for owner-authoritative components
    // of client-owned entities) equals every other peer's copy: within 2 Quantize steps, exact otherwise; Entity
    // fields through NetIds; the derived Transform of a child is not compared (its Parent and LocalTransform are)
    std::vector<World*> peers{w.get()};
    for (auto& c : clientWorlds) peers.push_back(c.get());
    const Registry& reg = w->registry();
    int reported        = 0;
    auto fail           = [&](const std::string& m)
    {
        if (reported++ < 30) check(false, "replicas differ: " + m);
    };
    const std::vector<NetId> ids = w->replication().netIds();
    for (size_t p = 1; p < peers.size(); ++p)
    {
        // A client must not keep replicated entities the host does not have
        for (NetId id : peers[p]->replication().netIds())
            // S25: a client keeps the state entity of a space the host unloaded (a bound NetId, no Despawn)
            if (!std::binary_search(ids.begin(), ids.end(), id) &&
                !(id >= kFirstSpaceStateNetId && id < kFirstDynamicNetId &&
                  id - kFirstSpaceStateNetId < w->spaceCount() && !w->spaceLoaded(SpaceId(id - kFirstSpaceStateNetId))))
                fail("peer " + std::to_string(p) + " has NetId " + std::to_string(id) + ", the host does not");
    }
    for (NetId id : ids)
    {
        const Entity he     = w->replication().entityOfNetId(id);
        const uint8_t owner = w->ownerOf(he);
        std::vector<Entity> handles{he};
        bool all = true;
        for (size_t p = 1; p < peers.size(); ++p)
        {
            handles.push_back(peers[p]->replication().entityOfNetId(id));
            if (!handles.back().valid())
            {
                fail("peer " + std::to_string(p) + " lacks NetId " + std::to_string(id));
                all = false;
            }
        }
        if (!all) continue;
        const std::string what = "NetId " + std::to_string(id) + " (host entity " + std::to_string(he.index) + ")";
        const bool child       = reg.entityMask(he).test(Parent::id);
        const bool state       = (reg.flags(he) & (Registry::kWorldState | Registry::kSpaceState)) != 0;
        // The parent through NetIds
        for (size_t p = 1; p < peers.size(); ++p)
        {
            const Parent* hp = w->registry().tryRead<Parent>(he);
            const Parent* cp = peers[p]->registry().tryRead<Parent>(handles[p]);
            const NetId a    = hp ? w->replication().netIdOf(hp->entity) : 0;
            const NetId b    = cp ? peers[p]->replication().netIdOf(cp->entity) : 0;
            if (a != b)
                fail("peer " + std::to_string(p) + ", " + what + ": the parent is NetId " + std::to_string(b) +
                     ", the host's " + std::to_string(a));
        }
        for (size_t c = 0; c < kMaxComponents; ++c)
        {
            const ComponentId cid               = ComponentId(c);
            const Registry::ComponentInfo& info = reg.componentInfo(cid);
            if (!info.registered || cid == Parent::id) continue;
            if (info.replicate != Replicate::Host && info.replicate != Replicate::HostReliable &&
                info.replicate != Replicate::OwnerAuthoritative)
                continue;
            if (cid == Transform::id && (child || state)) continue;
            size_t authority = 0;
            if (info.replicate == Replicate::OwnerAuthoritative && owner != 0)
            {
                if (size_t(owner) >= peers.size()) continue;
                authority = owner;
            }
            const World& aw = *peers[authority];
            const bool aHas = aw.registry().entityMask(handles[authority]).test(cid);
            std::vector<NetLeaf> a;
            if (aHas) a = aw.replication().flattenComponent(handles[authority], cid);
            for (size_t p = 0; p < peers.size(); ++p)
            {
                if (p == authority) continue;
                const bool has          = peers[p]->registry().entityMask(handles[p]).test(cid);
                const std::string where = "peer " + std::to_string(p) + ", " + what + ", " + info.name +
                                          " (authority peer " + std::to_string(authority) + ")";
                if (has != aHas)
                {
                    fail(where + ": " + (has ? "present" : "missing") + " on the copy");
                    continue;
                }
                if (!has) continue;
                const std::vector<NetLeaf> b = peers[p]->replication().flattenComponent(handles[p], cid);
                if (a.size() != b.size())
                {
                    fail(where + ": the fields differ in number");
                    continue;
                }
                for (size_t k = 0; k < a.size(); ++k)
                {
                    bool same;
                    if (a[k].step > 0)
                        same = std::abs(a[k].value - b[k].value) <= 2 * a[k].step * (1 + 1e-9) + 1e-12;
                    else
                        same = a[k].exact == b[k].exact;
                    if (!same)
                    {
                        std::string detail;
                        if (a[k].step > 0)
                            detail =
                                ": " + std::to_string(b[k].value) + " vs the authority's " + std::to_string(a[k].value);
                        fail(where + ", field " + a[k].name + detail);
                    }
                }
            }
        }
    }
}

}  // namespace Engine
}  // namespace Saiga
