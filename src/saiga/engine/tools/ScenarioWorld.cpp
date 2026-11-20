/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ScenarioWorld: the world of a scenario with its helpers, the step observation and the save/load swap.

#include "saiga/engine/tools/ScenarioWorld.h"

#include "saiga/engine/core/Log.h"
#include "saiga/engine/net/NetOwner.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/save/SaveDiff.h"

#include <algorithm>
#include <cmath>

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
    return s;
}
}  // namespace

ScenarioWorld::ScenarioWorld(const WorldSetup& s, const ScenarioOptions& o, const ScenarioNetwork& n)
    : setup(scenarioSetup(s)), options(o), network(n), asyncRng(hashCombine(s.seed, 0x5CE7A110ull))
{
    logStart = engineLog().lastSequence();
    if (network.enabled)
    {
        // The host (peer 0) and the clients over a loopback network without loss (§17.3); reproducible (R-DET-1)
        LoopbackOptions lo;
        lo.delayRounds = network.delaySteps;
        lo.seed        = hashCombine(setup.seed, 0x4E7ull);
        transports     = LoopbackTransport::createNetwork(1 + std::max(1, options.clients), lo);
        WorldSetup hs  = setup;
        hs.transport   = transports[0].get();
        hs.role        = NetRole::Host;
        hs.localPeer   = 0;
        w              = std::make_unique<World>(hs);
    }
    else
        w = std::make_unique<World>(setup);
    // A replacement world shares the data (also after reload_data)
    setup.sharedData = w->sharedDataStore();
    w->setTickEveryStep(true);
    if (network.enabled)
    {
        auto report = [this](const std::string& m)
        {
            if (failed.size() < 200) failed.push_back("step " + std::to_string(observedSteps) + ": " + m);
        };
        w->replication().setWriteDetector(report);
        for (size_t i = 1; i < transports.size(); ++i)
        {
            WorldSetup cs = setup;
            cs.transport  = transports[i].get();
            cs.role       = NetRole::Client;
            cs.localPeer  = uint8_t(i);
            clientWorlds.push_back(std::make_unique<World>(cs));
            clientWorlds.back()->setTickEveryStep(true);
            clientWorlds.back()->replication().setWriteDetector(report);
            clientSteps.push_back(0);
        }
    }
    attach();
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

ScenarioWorld::~ScenarioWorld()
{
    // The API detaches from the world before the world goes away
    apiPtr.reset();
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
    if (network.enabled) stepClients();
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
    auto fresh                       = std::make_unique<World>(setup);
    // Before the load: configure() and onLoad() of the loaded game objects see the scenario's tick setting
    fresh->setTickEveryStep(w->tickEveryStep());
    const LoadResult r = fresh->tryLoad(saved);
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
            synced = true;
            for (auto& c : clientWorlds) synced = synced && c->replication().snapshotSequence() >= needed;
        }
    }
    if (!check(synced, "the final network sync did not finish in 600 steps")) return;
    check(steps <= bound, "the final network sync took " + std::to_string(steps) + " steps (at most " +
                              std::to_string(bound) + " without loss)");
    compareReplicas();
}

void ScenarioWorld::compareReplicas()
{
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
            if (!std::binary_search(ids.begin(), ids.end(), id))
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
