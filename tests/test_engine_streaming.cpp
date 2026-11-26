/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Streaming (docs/engine_design.md §12.3, §12.4; S24): the configuration, the resident chunk set (synchronous and on
// the asynchronous pool with held jobs), edits of pending and non-resident chunks, the Simulated tag, the mover hold
// and the stalls, arrivals and islands, saves, aborted steps and clients. The worlds use tests/streaming_test_setup.h.
// The tests that count no generations, cache hits or jobs share one content cache (sharedStreamingCache: the
// StreamingConfig.*, Simulated.*, Arrival.* except HeldUntilIslandResident, StreamingSave.* and StreamingNet.* cases);
// the Residency.* cases that count jobs or generations use their own.

#include "saiga/engine/components/IslandState.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/components/SpaceLink.h"
#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/core/StreamingPool.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/save/SaveFormat.h"
#include "saiga/engine/streaming/Islands.h"
#include "saiga/engine/sync/ExtractSystem.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainStore.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/tools/ScenarioWorld.h"
#include "saiga/engine/world/Events.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>

#include "streaming_test_setup.h"
#include <condition_variable>

using namespace Saiga;
using namespace Saiga::Engine;
using namespace StreamingTest;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

// ---- a latch for the residency's jobs (file-static: the hook of a job never outlives it)
struct Latch
{
    std::mutex m;
    std::condition_variable cv;
    bool closed = false;
    int entered = 0;  // jobs that reached the latch since it was closed
    void close()
    {
        std::lock_guard<std::mutex> l(m);
        closed  = true;
        entered = 0;
    }
    int waiting()
    {
        std::lock_guard<std::mutex> l(m);
        return entered;
    }
    void open()
    {
        {
            std::lock_guard<std::mutex> l(m);
            closed = false;
        }
        cv.notify_all();
    }
    void wait()
    {
        std::unique_lock<std::mutex> l(m);
        if (closed) entered++;
        cv.wait(l, [this] { return !closed; });
    }
};
Latch& latch()
{
    static Latch l;
    return l;
}
// Closes the latch for its scope (declared after the world: opened before the world's pool joins its workers)
struct ScopedLatch
{
    ScopedLatch() { latch().close(); }
    ~ScopedLatch() { latch().open(); }
};
// Holds the jobs of the listed chunks (all: every job) on the latch
std::function<void(const ChunkId&)> holdChunks(std::set<ChunkId> chunks)
{
    return [chunks](const ChunkId& c)
    {
        if (chunks.empty() || chunks.count(c)) latch().wait();
    };
}

double seconds()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Waits (bounded) until n held jobs run on the pool's workers (so a later cancel or caller job cannot take them)
bool waitHeld(int n, double timeout = 10)
{
    const double end = seconds() + timeout;
    while (latch().waiting() < n)
    {
        if (seconds() > end) return false;
        std::this_thread::yield();
    }
    return true;
}

// Steps until pred holds (bounded by a deadline; the pool's workers run meanwhile)
bool stepUntil(World& w, const std::function<bool()>& pred, double timeout = 20)
{
    const double end = seconds() + timeout;
    while (!pred())
    {
        if (seconds() > end) return false;
        if (w.terrain().poolCreated()) w.terrain().streamingPool().waitIdleFor(0.002);
        w.step();
    }
    return true;
}

// The chunks within distance d of p in the space (brute force over the space)
std::set<ChunkId> chunksWithin(const World& w, SpaceId space, const Vec3& p, double d)
{
    std::set<ChunkId> out;
    const int32_t n = w.spaceTemplate(space).sizeChunks;
    for (int32_t z = 0; z < n; ++z)
        for (int32_t x = 0; x < n; ++x)
        {
            const double dx = std::max({double(x) * 256 - p.x(), 0.0, p.x() - double(x + 1) * 256});
            const double dz = std::max({double(z) * 256 - p.z(), 0.0, p.z() - double(z + 1) * 256});
            if (std::sqrt(dx * dx + dz * dz) <= d)
            {
                ChunkId c;
                c.space = space;
                c.x     = x;
                c.z     = z;
                out.insert(c);
            }
        }
    return out;
}

RegionId level0(const ChunkId& c)
{
    RegionId r;
    r.space = c.space;
    r.level = 0;
    r.x     = c.x;
    r.z     = c.z;
    return r;
}

// A Ticker; the world ticks every object every step (the counts test Simulated, not the tick rings)
Entity spawnTicker(World& w, const Vec3& p, SpaceId space = 0)
{
    w.setTickEveryStep(true);
    return w.spawnIn<Ticker>(space, p);
}

uint32_t updates(World& w, Entity e)
{
    return w.read<TickCount>(e).updates;
}

// A tree of the chunk's placement at least `away` metres from (x, z) (TombstoneOnEdit trees near an edit would be
// tombstoned by it)
std::optional<InstanceKey> treeAwayFrom(World& w, const ChunkId& c, double x, double z, double away)
{
    const auto p = w.generation().chunkPlacement(c, w.effectivePlacementVersion(c));
    for (const TierOneInstance& t : p->instances)
    {
        const double tx = double(c.x) * 256 + double(t.x) / 256, tz = double(c.z) * 256 + double(t.z) / 256;
        if (std::hypot(tx - x, tz - z) > away) return t.key;
    }
    return std::nullopt;
}

TerrainBrush addBrush(double x, double z, double radius = 8, double strength = 2)
{
    TerrainBrush b;
    b.op       = TerrainBrush::Add;
    b.center   = Vec3(x, 0, z);
    b.radius   = radius;
    b.strength = strength;
    return b;
}

// ---- phase-1 probe (BetweenStepChangeSeenNextStep): chunkReady of the first player in phase 1
std::vector<int> g_phase1Ready;
void phase1Probe(World& w, double)
{
    w.viewAll<const Player, const Transform>().each(
        [&](Entity, const Player&, const Transform& t)
        { g_phase1Ready.push_back(w.chunkReady(t.space(), t.position().x(), t.position().z()) ? 1 : 0); });
}
// Phase-4 spawns (NewEntitiesEvaluatedAtCreation): commands applied at the phase's flush
std::vector<Vec3> g_spawnRequests;
std::vector<Entity> g_spawned;
void spawnSystem(World& w, double)
{
    for (const Vec3& p : g_spawnRequests) g_spawned.push_back(w.spawn<Ticker>(p));
    g_spawnRequests.clear();
}
void probeSystems(SystemList& s)
{
    registerStreamingTestSystems(s);
    s.add(Phase::Input, "phase1_probe", phase1Probe, RunOn::Host);
    s.add(Phase::Gameplay, "spawn_probe", spawnSystem, RunOn::Host);
}

// ---- aborts
bool g_throwPhase7 = false, g_throwPhase8 = false;
void throwingStreaming(World&, double)
{
    if (g_throwPhase7) throw std::runtime_error("phase 7 test abort");
}
void throwingExtract(World&, double)
{
    if (g_throwPhase8) throw std::runtime_error("phase 8 test abort");
}
void abortSystems(SystemList& s)
{
    registerStreamingTestSystems(s);
    s.add(Phase::Streaming, "throw7", throwingStreaming, RunOn::Host);
    s.add(Phase::Extract, "throw8", throwingExtract, RunOn::Host);
}

// Steps n times and collects the IslandLimitEvents readable after each step (each event exactly once: an event of step
// k is readable between steps k + 1 and k + 2)
std::vector<uint32_t> stepCollectingIslandEvents(World& w, int n)
{
    std::vector<uint32_t> out;
    for (int i = 0; i < n; ++i)
    {
        w.step();
        for (const IslandLimitEvent& e : w.events.read<IslandLimitEvent>()) out.push_back(e.count);
    }
    return out;
}

// ---- brute force of G8 for every alive entity (an implementation of its own: the root's rule over every centre cell,
// descendants take their root's flag)
void expectFlagsMatchBruteForce(World& w, const char* when)
{
    const Registry& reg = w.registry();
    // The centre cells: the roots of Player (not held) and StreamingAnchor entities
    std::set<uint32_t> players, anchors;
    for (uint32_t i = 1; i < reg.slotCount(); ++i)
    {
        const Entity e = reg.aliveAt(i);
        if (!e.valid()) continue;
        if (reg.has<Player>(e)) players.insert(w.rootOf(e).index);
        if (reg.has<StreamingAnchor>(e)) anchors.insert(w.rootOf(e).index);
    }
    struct Cell
    {
        SpaceId space;
        int64_t x, z;
    };
    std::vector<Cell> centres;
    auto cellOf = [](double v) { return int64_t(std::floor(v / 32)); };
    for (const std::set<uint32_t>* set : {&players, &anchors})
        for (uint32_t i : *set)
        {
            const Entity r = reg.aliveAt(i);
            if (set == &players && w.heldRoot(r)) continue;
            const Transform& t = reg.read<Transform>(r);
            centres.push_back({t.space(), cellOf(t.position().x()), cellOf(t.position().z())});
        }
    auto rule = [&](Entity r)
    {
        if (reg.flags(r) & (Registry::kWorldState | Registry::kSpaceState)) return true;
        if (w.heldRoot(r)) return false;
        if (players.count(r.index) || anchors.count(r.index)) return true;
        const Transform& t = reg.read<Transform>(r);
        const double rad   = w.spaceTemplate(t.space()).simulationRadius;
        for (const Cell& c : centres)
        {
            const int64_t dx = cellOf(t.position().x()) - c.x, dz = cellOf(t.position().z()) - c.z;
            if (c.space == t.space() && double(dx * dx + dz * dz) * 32 * 32 <= rad * rad) return true;
        }
        const Behaviour* b = reg.tryRead<Behaviour>(r);
        return b && b->alwaysSimulate && w.chunkReady(t.space(), t.position().x(), t.position().z());
    };
    for (uint32_t i = 1; i < reg.slotCount(); ++i)
    {
        const Entity e = reg.aliveAt(i);
        if (!e.valid()) continue;
        const Entity r = w.rootOf(e);
        ASSERT_EQ(w.simulated(e), rule(r)) << when << ": entity " << e.index << " (root " << r.index << ")";
    }
}
}  // namespace

// ==== StreamingConfig ===============================================================================================

TEST(StreamingConfig, ValidationIsFatal)
{
    {
        WorldSetup s                                = streamingTestSetup(sharedStreamingCache());
        s.config.spaceTemplates[0].simulationRadius = 200;  // > fullRadius 128
        EXPECT_DEATH(World w(s), "overworld.*simulationRadius");
    }
    {
        WorldSetup s                            = streamingTestSetup(sharedStreamingCache());
        s.config.spaceTemplates[1].unloadMargin = -1;
        EXPECT_DEATH(World w(s), "test_valley.*unloadMargin");
    }
    {
        WorldSetup s                          = streamingTestSetup(sharedStreamingCache());
        s.config.spaceTemplates[0].fullRadius = std::nan("");
        EXPECT_DEATH(World w(s), "overworld.*fullRadius must be finite");
    }
}

TEST(StreamingConfig, OffKeepsS23bBehaviour)
{
    WorldSetup s       = streamingTestSetup(sharedStreamingCache());
    s.config.streaming = false;
    World w(s);
    for (const SystemList::System& sys : w.systemList().systems(Phase::Streaming))
    {
        EXPECT_NE(sys.name, "streaming");
        EXPECT_NE(sys.name, "simulation");
    }
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    const Entity far    = spawnTicker(w, Vec3(20000, 0, 20000));
    w.step(3);
    EXPECT_TRUE(w.simulated(far));
    EXPECT_GT(updates(w, far), 0u);
    EXPECT_TRUE(w.residency().chunkStatuses().empty());
    EXPECT_FALSE(w.terrain().poolCreated());
    EXPECT_TRUE(w.chunkReady(0, 20000, 20000));
    // An arrival completes in the next step (S21), nothing is held
    ASSERT_TRUE(w.requestArrival(player, 0, Vec3(9000, 0, 9000)));
    EXPECT_FALSE(w.heldRoot(player));
    w.step();
    EXPECT_FALSE(w.has<PendingTransition>(player));
    EXPECT_NEAR(w.read<Transform>(player).position().x(), 9000, 1e-9);
    // The mover is never held
    w.get<CharacterIntent>(player).desiredVelocity = Vec3(5, 0, 0);
    const double x0                                = w.read<Transform>(player).position().x();
    w.step();
    EXPECT_GT(w.read<Transform>(player).position().x(), x0);
    EXPECT_EQ(w.residency().stats().moverHolds, 0u);
}

TEST(StreamingConfig, ScenarioWorldLowersRadiiAndLoadsSynchronously)
{
    WorldSetup s = streamingTestSetup(sharedStreamingCache());
    for (SpaceTemplate& t : s.config.spaceTemplates)
    {
        t.fullRadius       = 1000;
        t.simulationRadius = 384;
        t.residencyRadius  = 600;
        t.prefetchMargin   = 256;
    }
    ScenarioWorld sw(s);
    const SpaceTemplate& t = sw.world().spaceTemplate(0);
    EXPECT_EQ(t.fullRadius, 128);
    EXPECT_EQ(t.simulationRadius, 128);
    EXPECT_EQ(t.residencyRadius, 128);
    EXPECT_EQ(t.prefetchMargin, 0);
    EXPECT_TRUE(sw.world().streamingSettings().synchronous);
    // The scenario distances (S24 Scope 1): required and requested 128 m, kept 192 m with the test setup's margin
    EXPECT_EQ(t.requestedDistance(), 128);
}

// ==== Residency =====================================================================================================

TEST(Residency, RequestSetFollowsPlayer)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    int unloads = 0;
    ResidencyListener l;
    l.onUnload = [&](const ChunkId&) { unloads++; };
    w.residency().addListener(l);
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    w.step();
    const SpaceTemplate& t = w.spaceTemplate(0);
    auto requestedNow      = [&](const Vec3& p)
    {
        const std::set<ChunkId> expected = chunksWithin(w, 0, p, t.requestedDistance());
        for (const ChunkId& c : expected)
        {
            EXPECT_TRUE(w.residency().requested(c));
            EXPECT_TRUE(w.residency().ready(c));
        }
        int requested = 0;
        for (const ChunkStatus& s : w.residency().chunkStatuses()) requested += s.requested ? 1 : 0;
        EXPECT_EQ(requested, int(expected.size()));
    };
    requestedNow(Vec3(1150, 0, 1400));
    // 100 m east: the chunks that left the requested distance but are kept stay resident, not ready
    w.get<Transform>(player).setPosition(Vec3(1250, 0, 1400));
    w.step();
    requestedNow(Vec3(1250, 0, 1400));
    const std::set<ChunkId> before = chunksWithin(w, 0, Vec3(1150, 0, 1400), t.requestedDistance());
    const std::set<ChunkId> kept   = chunksWithin(w, 0, Vec3(1250, 0, 1400), t.keptDistance());
    int hysteresis                 = 0;
    for (const ChunkId& c : before)
        if (!w.residency().requested(c) && kept.count(c))
        {
            EXPECT_EQ(w.residency().state(c), ChunkState::Resident);
            EXPECT_FALSE(w.residency().ready(c));
            hysteresis++;
        }
    EXPECT_GT(hysteresis, 0);
    // A further 1 km: beyond the kept distance every old chunk is None (tile unpinned, placement released)
    w.get<Transform>(player).setPosition(Vec3(5150, 0, 1400));
    w.step();
    for (const ChunkId& c : before)
    {
        EXPECT_EQ(w.residency().state(c), ChunkState::None);
        EXPECT_EQ(w.terrain().store(0).tile(level0(c)), nullptr);
        EXPECT_EQ(w.residency().heldPlacement(c, w.effectivePlacementVersion(c)), nullptr);
    }
    EXPECT_GE(unloads, int(before.size()));
    EXPECT_EQ(w.residency().stats().unloaded, uint64_t(unloads));
}

TEST(Residency, TrivialChunksNeedNoJobs)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    const SpaceId flat = spaceNamed(w, "test_flat"), dungeon = spaceNamed(w, "test_dungeon");
    ASSERT_LT(flat, w.spaceCount());
    ASSERT_LT(dungeon, w.spaceCount());
    addPlayer(w, Vec3(500, 0, 500), flat);
    addPlayer(w, Vec3(2100, 0, 2100), dungeon);
    w.step();
    EXPECT_TRUE(w.chunkReady(flat, 500, 500));
    EXPECT_TRUE(w.chunkReady(dungeon, 2100, 2100));
    EXPECT_FALSE(w.terrain().poolCreated());
    EXPECT_EQ(w.residency().stats().residentChunks, w.residency().stats().requestedChunks);
}

TEST(Residency, BetweenStepChangeSeenNextStep)
{
    WorldSetup s = streamingTestSetup(sharedStreamingCache());
    s.systems    = probeSystems;
    World w(s);
    w.setStreamingSynchronous(true);
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    w.step();
    // A teleport 1 km away between steps: phase 1 of the next step sees its chunk ready (the step-start refresh)
    w.get<Transform>(player).teleport(Vec3(5150, 0, 1400), Quat::Identity());
    g_phase1Ready.clear();
    w.step();
    ASSERT_EQ(g_phase1Ready.size(), 1u);
    EXPECT_EQ(g_phase1Ready[0], 1);
    // As a world loaded right after the teleport
    w.get<Transform>(player).teleport(Vec3(9150, 0, 1400), Quat::Identity());
    World loaded(s);
    loaded.setStreamingSynchronous(true);
    ASSERT_TRUE(loaded.tryLoad(w.save()).ok());
    EXPECT_TRUE(loaded.chunkReady(0, 9150, 1400));
    g_phase1Ready.clear();
    loaded.step();
    ASSERT_EQ(g_phase1Ready.size(), 1u);
    EXPECT_EQ(g_phase1Ready[0], 1);
}

TEST(Residency, AsyncAppliesInPhase7)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    const ChunkId c               = chunkOf(0, 1150, 1400);
    w.residency().jobHookForTests = holdChunks({c});
    ScopedLatch hold;
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    // The job is held through the step's drain (a cache hit could finish before it)
    w.step();
    EXPECT_EQ(w.residency().state(c), ChunkState::Requested);
    ASSERT_TRUE(waitHeld(1));
    latch().open();
    // Every job done and its completion drained between steps (an updateView-style drain): still Requested
    w.terrain().streamingPool().waitIdle();
    w.terrain().drainStreamingPool();
    EXPECT_EQ(w.residency().state(c), ChunkState::Requested);
    EXPECT_FALSE(w.chunkReady(0, 1150, 1400));
    w.step();
    EXPECT_EQ(w.residency().state(c), ChunkState::Resident);
    EXPECT_TRUE(w.chunkReady(0, 1150, 1400));
    (void)player;
}

TEST(Residency, StaleResultDropped)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    // One job in flight: only the held one is running when its chunk unloads (others would count their cancels too)
    StreamingSettings one;
    one.maxInFlight = 1;
    w.setStreamingSettings(one);
    const ChunkId c               = chunkOf(0, 1150, 1400);
    w.residency().jobHookForTests = holdChunks({c});
    const Entity player           = addPlayer(w, Vec3(1150, 0, 1400));
    ScopedLatch hold;
    w.step();
    ASSERT_EQ(w.residency().state(c), ChunkState::Requested);
    ASSERT_TRUE(waitHeld(1));
    // Unloaded (5 km away) and requested again: the held job's result carries the old serial
    w.get<Transform>(player).teleport(Vec3(5150, 0, 1400), Quat::Identity());
    w.step();
    EXPECT_EQ(w.residency().state(c), ChunkState::None);
    w.get<Transform>(player).teleport(Vec3(1150, 0, 1400), Quat::Identity());
    w.step();
    EXPECT_EQ(w.residency().state(c), ChunkState::Requested);
    w.residency().jobHookForTests = nullptr;
    latch().open();
    ASSERT_TRUE(stepUntil(w, [&] { return w.residency().ready(c); }));
    EXPECT_EQ(w.residency().stats().resultsDropped, 1u);
}

// A v2 world with the v1 pins of a loaded save (placement version 1 by default, 2 can be injected)
void loadV1Pins(World& w)
{
    // test_terrain v1 has another generator identity: its own cache (once per process)
    static std::shared_ptr<ContentCache> v1Cache = std::make_shared<ContentCache>();
    World v1(streamingTestSetup(v1Cache, 1));
    ASSERT_TRUE(w.tryLoad(v1.save()).ok());
    // Pinned at 1, the current placement version 2: a record can name version 1
    ASSERT_EQ(w.generation().space(0).pinned, 1);
    ASSERT_EQ(w.effectivePlacementVersion(chunkOf(0, 1150, 1400)), 2);
}

TEST(Residency, PlacementVersionChangeReRequests)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    loadV1Pins(w);
    const ChunkId c               = chunkOf(0, 1150, 1400);
    w.residency().jobHookForTests = holdChunks({c});
    addPlayer(w, Vec3(1150, 0, 1400));
    ScopedLatch hold;
    w.step();
    ASSERT_EQ(w.residency().state(c), ChunkState::Requested);
    ASSERT_TRUE(waitHeld(1));
    // Another placement version for the pending chunk (a load or a record message would do this): the held result is
    // dropped, the new request uses the new version
    w.contentRecordsForUpdate().injectForTests(c, 1, {});
    w.onRecordChanged(c);
    w.residency().jobHookForTests = nullptr;
    latch().open();
    ASSERT_TRUE(stepUntil(w, [&] { return w.residency().ready(c); }));
    // The released old job may report after the new request's result applied (two workers): wait for its drop (a CI
    // race before)
    ASSERT_TRUE(stepUntil(w, [&] { return w.residency().stats().resultsDropped >= 1; }));
    EXPECT_EQ(w.residency().stats().resultsDropped, 1u);
    EXPECT_NE(w.residency().heldPlacement(c, 1), nullptr);
}

TEST(Residency, ResidentVersionChange)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    loadV1Pins(w);
    w.setStreamingSynchronous(true);
    // A chunk without a site (a stamped chunk has a record of the current version)
    addPlayer(w, Vec3(5150, 0, 1400));
    w.step();
    const ChunkId c = chunkOf(0, 5150, 1400);
    ASSERT_TRUE(w.residency().ready(c));
    ASSERT_EQ(w.chunkRecord(c), nullptr);
    w.setStreamingSynchronous(false);
    w.residency().jobHookForTests = holdChunks({c});
    ScopedLatch hold;
    w.contentRecordsForUpdate().injectForTests(c, 1, {});
    w.onRecordChanged(c);
    w.step();
    // Not ready, the placement requested at the new version, the base pinned
    EXPECT_FALSE(w.residency().ready(c));
    EXPECT_EQ(w.residency().partsReady(c), uint8_t(PartBase));
    ASSERT_NE(w.terrain().store(0).tile(level0(c)), nullptr);
    EXPECT_TRUE(w.terrain().store(0).tile(level0(c))->pinned);
    w.residency().jobHookForTests = nullptr;
    latch().open();
    ASSERT_TRUE(stepUntil(w, [&] { return w.residency().ready(c); }));
    EXPECT_NE(w.residency().heldPlacement(c, 1), nullptr);
}

// The acceptance test (asynchronous, a held result): an edit and a tombstone on a pending chunk survive its result
TEST(Residency, PendingChunkEditSurvives)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    const ChunkId c               = chunkOf(0, 1150, 1400);
    w.residency().jobHookForTests = holdChunks({c});
    const Entity player           = addPlayer(w, Vec3(1150, 0, 1400));
    ScopedLatch hold;
    w.step();
    ASSERT_EQ(w.residency().state(c), ChunkState::Requested);
    // The step counts a stall and the mover holds the player
    const uint64_t stalls                          = w.residency().stats().streamingStalls;
    w.get<CharacterIntent>(player).desiredVelocity = Vec3(5, 0, 0);
    const Vec3 p0                                  = w.read<Transform>(player).position();
    w.step();
    EXPECT_GT(w.residency().stats().streamingStalls, stalls);
    EXPECT_EQ(w.read<Transform>(player).position(), p0);
    EXPECT_GT(w.residency().stats().moverHolds, 0u);
    // An edit (8 m) and a tombstone of a tree outside its box
    const double ex = 1170, ez = 1420;
    const auto key = treeAwayFrom(w, c, ex, ez, 40);
    ASSERT_TRUE(key);
    const uint32_t seq0 = w.terrainEdits().sequence(c);
    w.editTerrain(0, addBrush(ex, ez));
    w.tombstone(c, *key);
    // The edit's own step, the tombstone's and those of the TombstoneOnEdit trees inside the brush (S23b)
    const uint32_t seq1 = w.terrainEdits().sequence(c);
    EXPECT_GE(seq1, seq0 + 2);
    const int64_t si = int64_t(ex / 2), sk = int64_t(ez / 2);
    const int32_t edited = w.terrainSample(0, si, sk);
    EXPECT_NE(edited, w.terrain().sample(0, si, sk));
    // The latch opens: the chunk becomes resident in a phase 7
    w.residency().jobHookForTests = nullptr;
    latch().open();
    ASSERT_TRUE(stepUntil(w, [&] { return w.residency().ready(c); }));
    const TerrainStore::Tile* tile = w.terrain().store(0).tile(level0(c));
    ASSERT_NE(tile, nullptr);
    ASSERT_TRUE(tile->pinned);
    // The pinned tile holds the base heights (the height source), the samples still show the edit
    const int64_t n1 = 128, ti = si - int64_t(c.x) * n1, tk = sk - int64_t(c.z) * n1;
    EXPECT_EQ((*tile->data.heights)[size_t(tk * (n1 + 1) + ti)], w.generation().height(0, si * 2, sk * 2));
    EXPECT_EQ(w.terrainSample(0, si, sk), edited);
    EXPECT_EQ(w.terrainEdits().sequence(c), seq1);
    EXPECT_EQ(w.residency().stats().appliedAfterModification, 1u);
    EXPECT_TRUE(w.isTombstoned(c, *key));
    // The stalls stop and the player moves again
    const uint64_t stallsNow = w.residency().stats().streamingStalls;
    w.step();
    EXPECT_EQ(w.residency().stats().streamingStalls, stallsNow);
    EXPECT_GT(w.read<Transform>(player).position().x(), p0.x());
}

// The acceptance test: an edit and a tombstone of a chunk 10 km from every centre are in its modification state
TEST(Residency, NonResidentEditInModificationState)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    w.step();
    const double ex = 13150, ez = 1400;
    const ChunkId c = chunkOf(0, ex, ez);
    const auto key  = treeAwayFrom(w, c, ex, ez, 40);
    ASSERT_TRUE(key);
    const uint64_t hooks0      = w.terrain().baseHookSamples();
    const uint64_t placements0 = w.generation().cache().stats().placementGenerations;
    w.editTerrain(0, addBrush(ex, ez));
    w.tombstone(c, *key);
    // The edit's step, the tombstone's and those of the TombstoneOnEdit trees inside the brush
    const uint32_t seq = w.terrainEdits().sequence(c);
    EXPECT_GE(seq, 2u);
    EXPECT_TRUE(w.terrainEdits().hasDeltas(c));
    ASSERT_NE(w.chunkRecord(c), nullptr);
    EXPECT_TRUE(w.isTombstoned(c, *key));
    EXPECT_EQ(w.residency().state(c), ChunkState::None);
    EXPECT_GT(w.terrain().baseHookSamples(), hooks0);
    EXPECT_GE(w.generation().cache().stats().placementGenerations, placements0);
    const int64_t si = int64_t(ex / 2), sk = int64_t(ez / 2);
    const int32_t edited = w.terrainSample(0, si, sk);
    // A save / load round trip keeps them
    World loaded(streamingTestSetup(w.generation().sharedCache()));
    loaded.setStreamingSynchronous(true);
    ASSERT_TRUE(loaded.tryLoad(w.save()).ok());
    EXPECT_EQ(loaded.terrainEdits().sequence(c), seq);
    EXPECT_TRUE(loaded.isTombstoned(c, *key));
    // An arrival there: the chunk is resident and shows the edit
    ASSERT_TRUE(loaded.requestArrival(player, 0, Vec3(ex, 0, ez)));
    loaded.step(2);
    EXPECT_FALSE(loaded.has<PendingTransition>(player));
    EXPECT_TRUE(loaded.residency().ready(c));
    EXPECT_EQ(loaded.terrainSample(0, si, sk), edited);
    EXPECT_TRUE(loaded.isTombstoned(c, *key));
}

TEST(Residency, CompleteResidency)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    addPlayer(w, Vec3(1150, 0, 1400));
    EXPECT_TRUE(w.completeResidency(30));
    for (const ChunkId& c : chunksWithin(w, 0, Vec3(1150, 0, 1400), w.spaceTemplate(0).requiredDistance()))
        EXPECT_TRUE(w.residency().ready(c));
    // A held job and a short timeout: false, what was ready is applied
    const ChunkId far             = chunkOf(0, 5150, 1400);
    w.residency().jobHookForTests = holdChunks({far});
    ScopedLatch hold;
    addPlayer(w, Vec3(5150, 0, 1400));
    // The held job runs on a worker (the caller would otherwise run it itself and block)
    w.step();
    ASSERT_TRUE(waitHeld(1));
    const double t0 = seconds();
    EXPECT_FALSE(w.completeResidency(0.05));
    EXPECT_LT(seconds() - t0, 5.0);
    EXPECT_FALSE(w.residency().ready(far));
}

TEST(Residency, ResidentTilesServeSamples)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    addPlayer(w, Vec3(1150, 0, 1400));
    w.step();
    const uint64_t h0 = w.terrain().baseHookSamples();
    for (int i = 0; i < 50; ++i) w.terrainHeight(0, 1100 + i, 1390 + i);
    EXPECT_EQ(w.terrain().baseHookSamples(), h0);
    w.terrainHeight(0, 15000, 15000);
    EXPECT_EQ(w.terrain().baseHookSamples(), h0 + 3);
}

TEST(Residency, VisualKeepsPinnedTiles)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    TerrainViewSettings v = w.terrain().viewSettings();
    v.tilesPerLevel       = 4;
    v.grassRadius         = 0;
    // A small range: few tiles per view (each visual tile is generated, test speed)
    v.residencyRange0 = 64;
    w.terrain().setViewSettings(v);
    const SpaceId s = spaceNamed(w, "test_small");
    addPlayer(w, Vec3(1150, 0, 1400), s);
    w.step();
    const int pinned = w.terrain().store(s).pinnedTiles();
    ASSERT_GT(pinned, 4);
    // The camera at the player, then far away: the visual path evicts no pinned tile (the visual budget counts the
    // visual tiles only)
    w.terrain().completeView(s, Vec3(1150, w.terrainHeight(s, 1150, 1400) + 10, 1400), 30);
    const size_t visualAtPlayer = sink.terrainTiles().size();
    EXPECT_GT(visualAtPlayer, 0u);
    w.terrain().completeView(s, Vec3(3800, 300, 3800), 30);
    EXPECT_EQ(w.terrain().store(s).pinnedTiles(), pinned);
    for (const ChunkId& c : chunksWithin(w, s, Vec3(1150, 0, 1400), 128))
        EXPECT_NE(w.terrain().store(s).tile(level0(c)), nullptr);
    EXPECT_LE(w.terrain().store(s).tilesAtLevel(0), 4);
    w.setRenderSink(nullptr);
}

// Blocker regression (plan review): a pinned, once-sent level-0 tile stayed sent after it left the desired set and
// overflowed the atlas
TEST(Residency, PinnedTilesDoNotOverflowAtlas)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    TerrainViewSettings v = w.terrain().viewSettings();
    v.tilesPerLevel       = 4;
    v.grassRadius         = 0;
    v.residencyRange0     = 100;
    w.terrain().setViewSettings(v);
    // The player keeps a block of chunks pinned; the camera sweeps across 12 of them (one per step)
    const SpaceId s = spaceNamed(w, "test_small");
    addPlayer(w, Vec3(1150, 0, 1400), s);
    w.step();
    std::set<RegionId> everSent;
    for (int k = 0; k < 12; ++k)
    {
        const Vec3 p(1150 - 256 + 256.0 * (k % 4), 0, 1400 - 128 + 256.0 * (k / 4));
        w.step();
        w.terrain().completeView(s, Vec3(p.x(), w.terrainHeight(s, p.x(), p.z()) + 20, p.z()), 30);
        int level0Sent = 0;
        for (const auto& t : sink.terrainTiles())
            if (t.first.level == 0)
            {
                level0Sent++;
                everSent.insert(t.first);
            }
        ASSERT_LE(level0Sent, 4) << "step " << k;
    }
    // More pinned tiles than the budget; a pinned tile that left the desired set is not in the sink but in the store
    EXPECT_GT(w.terrain().store(s).pinnedTiles(), 4);
    int leftSinkStillPinned = 0;
    for (const RegionId& r : everSent)
    {
        const TerrainStore::Tile* t = w.terrain().store(s).tile(r);
        if (!sink.terrainTiles().count(r) && t && t->pinned) leftSinkStillPinned++;
    }
    EXPECT_GT(leftSinkStillPinned, 0);
    w.setRenderSink(nullptr);
}

TEST(Residency, ViewSwitchKeepsPinnedTiles)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    TerrainViewSettings v = w.terrain().viewSettings();
    v.grassRadius         = 0;
    v.residencyRange0     = 64;
    w.terrain().setViewSettings(v);
    const SpaceId s = spaceNamed(w, "test_small");
    addPlayer(w, Vec3(1150, 0, 1400), s);
    w.step();
    w.terrain().completeView(s, Vec3(1150, w.terrainHeight(s, 1150, 1400) + 10, 1400), 30);
    const int pinned = w.terrain().store(s).pinnedTiles();
    ASSERT_GT(pinned, 0);
    // The view switches to another space (test_flat: its switch drops the old view space's streamed data the same way
    // as a terrain space's, without generating a second terrain)
    w.terrain().completeView(spaceNamed(w, "test_flat"), Vec3(500, 200, 500), 30);
    EXPECT_EQ(w.terrain().store(s).pinnedTiles(), pinned);
    EXPECT_EQ(w.terrain().store(s).tilesAtLevel(0), 0);
    const uint64_t h0 = w.terrain().baseHookSamples();
    w.terrainHeight(s, 1150, 1400);
    EXPECT_EQ(w.terrain().baseHookSamples(), h0);
    w.setRenderSink(nullptr);
}

TEST(Residency, PriorityRequiredFirst)
{
    // The hook's state before the World: the World joins its workers first
    std::mutex m;
    std::vector<ChunkId> order;
    World w(streamingTestSetup(sharedStreamingCache()));
    // One job in flight: the jobs start in the residency's submission order, so the hook order is the priority order
    // (with two the workers could log them in either order: a flaky failure before)
    StreamingSettings s;
    s.maxInFlight = 1;
    w.setStreamingSettings(s);
    w.residency().jobHookForTests = [&](const ChunkId& c)
    {
        std::lock_guard<std::mutex> l(m);
        order.push_back(c);
    };
    const Vec3 p(1150, 0, 1400);
    addPlayer(w, p);
    ASSERT_TRUE(stepUntil(w,
                          [&]
                          {
                              for (const ChunkId& c : chunksWithin(w, 0, p, w.spaceTemplate(0).requestedDistance()))
                                  if (!w.residency().ready(c)) return false;
                              return true;
                          }));
    std::lock_guard<std::mutex> l(m);
    const double R      = w.spaceTemplate(0).requiredDistance();
    bool prefetchSeen   = false;
    double lastRequired = -1;
    for (const ChunkId& c : order)
    {
        const double dx = std::max({double(c.x) * 256 - p.x(), 0.0, p.x() - double(c.x + 1) * 256});
        const double dz = std::max({double(c.z) * 256 - p.z(), 0.0, p.z() - double(c.z + 1) * 256});
        const double d  = std::sqrt(dx * dx + dz * dz);
        if (d <= R)
        {
            EXPECT_FALSE(prefetchSeen) << "a required chunk after a prefetch chunk";
            // Nearest first (by the integer distance of the priority)
            EXPECT_GE(int(d), int(lastRequired));
            lastRequired = d;
        }
        else
            prefetchSeen = true;
    }
}

TEST(Residency, PromotedPrefetchResubmitted)
{
    // The hook's state before the World (the World joins its workers first)
    std::mutex m;
    std::vector<ChunkId> order;
    World w(streamingTestSetup(sharedStreamingCache()));
    StreamingSettings s;
    s.maxInFlight = 64;
    w.setStreamingSettings(s);
    // Every job waits: the two workers hold two, the rest stay queued
    w.residency().jobHookForTests = [&](const ChunkId& c)
    {
        {
            std::lock_guard<std::mutex> l(m);
            order.push_back(c);
        }
        latch().wait();
    };
    ScopedLatch hold;
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    w.step();
    const int inFlight = w.residency().stats().pendingJobs;
    ASSERT_GT(inFlight, 4);
    // Toward a prefetch chunk: it becomes required, its queued job is cancelled and resubmitted (the count stays)
    w.get<Transform>(player).setPosition(Vec3(1150 + 150, 0, 1400));
    w.step();
    EXPECT_LE(w.residency().stats().pendingJobs, inFlight + 16);
    latch().open();
    ASSERT_TRUE(stepUntil(w, [&] { return w.residency().ready(chunkOf(0, 1300, 1400)); }));
    ASSERT_TRUE(stepUntil(w, [&] { return w.residency().stats().pendingJobs == 0; }));
}

// Regression: a running prefetch job whose chunk becomes required was "cancelled" by the promotion (the pool only flags
// a running job) and its result dropped, which left the chunk Requested for ever.
TEST(Residency, PromotedRunningPrefetchApplies)
{
    // The hook's state before the World (the World joins its workers first)
    std::atomic<bool> running{false};
    World w(streamingTestSetup(sharedStreamingCache()));
    const ChunkId target          = chunkOf(0, 1300, 1400);  // 130 m from the player: a prefetch chunk
    w.residency().jobHookForTests = [&](const ChunkId& c)
    {
        if (c != target) return;
        running = true;
        latch().wait();
    };
    ScopedLatch hold;
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    ASSERT_TRUE(stepUntil(w, [&] { return running.load(); }));
    ASSERT_EQ(w.residency().state(target), ChunkState::Requested);
    // The target's job is running (held on the latch); moving 20 m east makes the chunk required
    w.get<Transform>(player).setPosition(Vec3(1170, 0, 1400));
    w.step();
    latch().open();
    ASSERT_TRUE(stepUntil(w, [&] { return w.residency().ready(target); }));
    EXPECT_TRUE(w.completeResidency(10));
}

TEST(Residency, OutOfRangeClamped)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    const Entity a    = addPlayer(w, Vec3(-5, 0, 1400));
    const double edge = w.spaceTemplate(0).sizeMeters();
    const Entity b    = addPlayer(w, Vec3(edge + 50, 0, 1400));
    w.step();
    EXPECT_TRUE(w.chunkReady(0, -5, 1400));
    EXPECT_TRUE(w.chunkReady(0, edge + 50, 1400));
    w.get<CharacterIntent>(a).desiredVelocity = Vec3(-1, 0, 0);
    w.get<CharacterIntent>(b).desiredVelocity = Vec3(1, 0, 0);
    w.step();
    EXPECT_EQ(w.residency().stats().moverHolds, 0u);
    EXPECT_LT(w.read<Transform>(a).position().x(), -5);
}

TEST(Residency, StampOnResident)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    // Site 1 at (1536, 1536) across chunks (5..6, 5..6): one of its chunks with stamps; the player next to it
    ChunkId site = chunkOf(0, 1500, 1500);
    for (int z = 5; z <= 6; ++z)
        for (int x = 5; x <= 6; ++x)
            if (!w.generation().stampList(chunkOf(0, x * 256 + 1, z * 256 + 1)).empty())
                site = chunkOf(0, x * 256 + 1, z * 256 + 1);
    ASSERT_FALSE(w.generation().stampList(site).empty());
    w.residency().jobHookForTests = holdChunks({site});
    ScopedLatch hold;
    addPlayer(w, Vec3(1500, 0, 1500));
    ASSERT_TRUE(stepUntil(w, [&] { return w.residency().stats().pendingJobs <= 1; }));
    w.step(2);
    EXPECT_FALSE(w.residency().ready(site));
    EXPECT_FALSE(w.contentRecords().isStamped(site));
    w.residency().jobHookForTests = nullptr;
    latch().open();
    bool stampedWhenReady = false;
    ASSERT_TRUE(stepUntil(w,
                          [&]
                          {
                              if (w.residency().ready(site))
                              {
                                  // Stamped in the phase 7 in which it became resident
                                  stampedWhenReady = w.contentRecords().isStamped(site);
                                  return true;
                              }
                              return false;
                          }));
    EXPECT_TRUE(stampedWhenReady);
}

TEST(Residency, StallsCounted)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    const ChunkId a = chunkOf(0, 1150, 1400), b = chunkOf(0, 9150, 1400);
    w.residency().jobHookForTests = holdChunks({a, b});
    ScopedLatch hold;
    addPlayer(w, Vec3(1150, 0, 1400));
    addPlayer(w, Vec3(9150, 0, 1400));
    w.step();
    const StreamingStats s0 = w.residency().stats();
    w.step(3);
    EXPECT_EQ(w.residency().stats().streamingStalls, s0.streamingStalls + 3);
    EXPECT_EQ(w.residency().stats().playerStallSteps, s0.playerStallSteps + 6);
    EXPECT_EQ(w.residency().stalledPlayers(), 2);
}

// ==== Simulated =====================================================================================================

TEST(Simulated, FarEntitiesFreeze)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    addPlayer(w, Vec3(1150, 0, 1400));
    const Entity nearT = spawnTicker(w, Vec3(1180, 0, 1400));
    const Entity farT  = spawnTicker(w, Vec3(1300, 0, 1400));
    w.step(4);
    EXPECT_TRUE(w.simulated(nearT));
    EXPECT_FALSE(w.simulated(farT));
    EXPECT_GT(updates(w, nearT), 0u);
    EXPECT_GT(w.read<TickCount>(nearT).systemSteps, 0u);
    EXPECT_EQ(updates(w, farT), 0u);
    EXPECT_EQ(w.read<TickCount>(farT).systemSteps, 0u);
    // viewAll sees both in phase 4; ownedBy skips the far one; no event reaches it (behaviourEventSystem)
    int all = 0, owned = 0;
    w.registry().view<const TickCount>(0).each([&](Entity, const TickCount&) { all++; });
    EXPECT_EQ(all, 2);
    (void)owned;
    w.events.emit(DamageEvent{farT, Entity{}, 1.0});
    w.step();
    EXPECT_EQ(updates(w, farT), 0u);
}

TEST(Simulated, MatchesBruteForce)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    // The other space: test_small (2 m; test_valley's 1 m chunks cost four times as much)
    const SpaceId valley = spaceNamed(w, "test_small");
    const Entity a       = addPlayer(w, Vec3(1150, 0, 1400));
    const Entity b       = addPlayer(w, Vec3(1600, 0, 1300));
    Rng rng(7);
    std::vector<Entity> drifters;
    for (int i = 0; i < 400; ++i)
        drifters.push_back(w.spawn<Drifter>(Vec3(1000 + rng.uniform() * 800, 0, 1200 + rng.uniform() * 500)));
    // A few children (parented NPCs)
    for (int i = 0; i < 10; ++i)
    {
        const Entity c = w.create();
        w.add<Parent>(c, Parent{drifters[size_t(i)]});
    }
    for (int step = 0; step < 200; ++step)
    {
        // The players walk (a seeded walk); a changes space twice
        w.get<Transform>(a).setPosition(w.read<Transform>(a).position() + Vec3(rng.uniform() * 6 - 3, 0, 2));
        w.get<Transform>(b).setPosition(w.read<Transform>(b).position() + Vec3(-2, 0, rng.uniform() * 6 - 3));
        if (step == 60) w.moveToSpace(a, valley, Vec3(1150, 0, 1400), Quat::Identity());
        if (step == 120) w.moveToSpace(a, 0, Vec3(1300, 0, 1400), Quat::Identity());
        if (step % 25 == 5)
        {
            // Spawns, destroys, teleports and parent changes between steps
            w.spawn<Drifter>(Vec3(1100 + rng.uniform() * 600, 0, 1300 + rng.uniform() * 300));
            w.destroy(drifters[size_t(20 + step / 25)]);
            w.get<Transform>(drifters[size_t(40 + step / 25)])
                .teleport(Vec3(1150 + rng.uniform() * 50, 0, 1400), Quat::Identity());
            const Entity child = drifters[size_t(60 + step / 25)];
            if (!w.has<Parent>(child)) w.add<Parent>(child, Parent{drifters[size_t(80 + step / 25)]});
        }
        w.step();
        expectFlagsMatchBruteForce(w, ("step " + std::to_string(step)).c_str());
        if (::testing::Test::HasFatalFailure()) return;
    }
}

TEST(Simulated, AlwaysSimulateWhileRequestedAndResident)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    const Entity player                = addPlayer(w, Vec3(1150, 0, 1400));
    const Entity t                     = spawnTicker(w, Vec3(1150 + 150, 0, 1400));
    w.get<Behaviour>(t).alwaysSimulate = true;
    w.refreshSimulated(t);
    w.step(3);
    EXPECT_TRUE(w.simulated(t));
    EXPECT_GT(updates(w, t), 0u);
    // The player leaves: the chunk is kept by the hysteresis but not requested: frozen
    w.get<Transform>(player).setPosition(Vec3(1150 - 250, 0, 1400));
    w.step();
    const ChunkId c = chunkOf(0, 1300, 1400);
    ASSERT_FALSE(w.residency().requested(c));
    EXPECT_FALSE(w.simulated(t));
    // An alwaysSimulate edit through set_component applies without a move: covered by the world API test; here the
    // refresh: the flag off keeps it frozen, a player back near it simulates it
    w.get<Transform>(player).setPosition(Vec3(1150, 0, 1400));
    w.step();
    EXPECT_TRUE(w.simulated(t));
    w.get<Behaviour>(t).alwaysSimulate = false;
    w.refreshSimulated(t);
    EXPECT_FALSE(w.simulated(t));
}

TEST(Simulated, PlayersAndStateEntities)
{
    WorldSetup s = streamingTestSetup(sharedStreamingCache());
    World w(s);
    w.setStreamingSynchronous(true);
    const Entity a = addPlayer(w, Vec3(1150, 0, 1400));
    const Entity b = addPlayer(w, Vec3(13150, 0, 1400));
    w.step();
    EXPECT_TRUE(w.simulated(a));
    EXPECT_TRUE(w.simulated(b));
    // The world-state entity (IslandState) of a world without players in its space; the space-state entities exist with
    // a registered space-state component (the environment's Weather)
    EXPECT_TRUE(w.simulated(w.worldStateEntity()));
    WorldSetup env          = s;
    env.environment.enabled = true;
    World e(env);
    e.setStreamingSynchronous(true);
    addPlayer(e, Vec3(1150, 0, 1400));
    e.step();
    for (size_t i = 0; i < e.spaceCount(); ++i)
    {
        ASSERT_TRUE(e.spaceStateEntity(SpaceId(i)).valid());
        EXPECT_TRUE(e.simulated(e.spaceStateEntity(SpaceId(i))));
    }
}

TEST(Simulated, ChildrenFollowRoot)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    addPlayer(w, Vec3(1150, 0, 1400));
    const Entity far   = spawnTicker(w, Vec3(1350, 0, 1400));
    const Entity nearT = spawnTicker(w, Vec3(1160, 0, 1400));
    const Entity child = w.create();
    w.add<Parent>(child, Parent{far});
    w.step();
    EXPECT_FALSE(w.simulated(far));
    EXPECT_FALSE(w.simulated(child));
    w.remove<Parent>(child);
    w.add<Parent>(child, Parent{nearT});
    w.step();
    EXPECT_TRUE(w.simulated(child));
}

TEST(Simulated, NewEntitiesEvaluatedAtCreation)
{
    WorldSetup s = streamingTestSetup(sharedStreamingCache());
    s.systems    = probeSystems;
    World w(s);
    w.setStreamingSynchronous(true);
    addPlayer(w, Vec3(1150, 0, 1400));
    w.step();
    // Between steps
    const Entity far = spawnTicker(w, Vec3(1350, 0, 1400));
    w.step(3);
    EXPECT_EQ(updates(w, far), 0u);
    // By a command in phase 4 (flushed before phase 5): the far one never ticks, the near one from the next step
    g_spawned.clear();
    g_spawnRequests = {Vec3(1360, 0, 1400), Vec3(1155, 0, 1400)};
    w.step();
    ASSERT_EQ(g_spawned.size(), 2u);
    EXPECT_FALSE(w.simulated(g_spawned[0]));
    EXPECT_TRUE(w.simulated(g_spawned[1]));
    w.step(3);
    EXPECT_EQ(updates(w, g_spawned[0]), 0u);
    EXPECT_GT(updates(w, g_spawned[1]), 0u);
}

TEST(Simulated, AnchorIsCentre)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    const Entity anchor = w.spawn<Ticker>(Vec3(5150, 0, 1400));
    w.add<StreamingAnchor>(anchor);
    w.step();
    EXPECT_TRUE(w.chunkReady(0, 5150, 1400));
    EXPECT_TRUE(w.simulated(anchor));
    // A Player child of a moving root: a centre at its world position, its root simulated
    const Entity vehicle = w.spawn<Ticker>(Vec3(9150, 0, 1400));
    const Entity driver  = w.create();
    w.add<Player>(driver);
    w.add<Parent>(driver, Parent{vehicle});
    w.step();
    EXPECT_TRUE(w.chunkReady(0, 9150, 1400));
    EXPECT_TRUE(w.simulated(vehicle));
    EXPECT_TRUE(w.simulated(driver));
    w.get<Transform>(vehicle).setPosition(Vec3(13150, 0, 1400));
    w.step(2);
    EXPECT_TRUE(w.chunkReady(0, 13150, 1400));
}

TEST(Simulated, UnfreezeContinuesSinceTick)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    const Entity t      = spawnTicker(w, Vec3(1350, 0, 1400));
    w.step(5);
    EXPECT_FALSE(w.simulated(t));
    const double since = w.read<Behaviour>(t).sinceTick;
    w.step(5);
    EXPECT_EQ(w.read<Behaviour>(t).sinceTick, since);
    w.get<Transform>(player).setPosition(Vec3(1340, 0, 1400));
    w.step(3);
    EXPECT_TRUE(w.simulated(t));
    EXPECT_GT(updates(w, t), 0u);
}

TEST(Simulated, MoverHolds)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    const ChunkId held            = chunkOf(0, 1300, 1400);
    w.residency().jobHookForTests = holdChunks({held});
    ScopedLatch hold;
    // The player near the border (the drifter must be simulated to move)
    addPlayer(w, Vec3(1260, 0, 1400));
    ASSERT_TRUE(stepUntil(w, [&] { return w.chunkReady(0, 1260, 1400); }));
    const Entity d = w.spawn<Drifter>(Vec3(1270, 0, 1400));
    w.remove<Behaviour>(d);
    w.get<CharacterIntent>(d).desiredVelocity = Vec3(30, 0, 0);
    for (int i = 0; i < 60; ++i) w.step();
    // Stops at the border of the held chunk (x = 1280)
    EXPECT_LT(w.read<Transform>(d).position().x(), 1280.0);
    EXPECT_GT(w.read<Transform>(d).position().x(), 1270.0);
    EXPECT_GT(w.residency().stats().moverHolds, 0u);
}

// ==== Arrival =======================================================================================================

TEST(Arrival, HeldUntilIslandResident)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    w.setRenderMesh(player, "test/chest");
    ASSERT_TRUE(w.completeResidency(30));
    w.step();
    ASSERT_TRUE(w.extract().instanceOf(player));
    const Vec3 target(5150, 0, 1400);
    w.residency().jobHookForTests = holdChunks({});
    ScopedLatch hold;
    const Vec3 pose0 = w.read<Transform>(player).position();
    ASSERT_TRUE(w.requestArrival(player, 0, target));
    EXPECT_TRUE(w.has<PendingTransition>(player));
    EXPECT_FALSE(w.simulated(player));
    w.step(3);
    EXPECT_FALSE(w.extract().instanceOf(player));
    EXPECT_EQ(w.read<Transform>(player).position(), pose0);
    ASSERT_TRUE(w.arrivalProgress(player));
    EXPECT_LT(*w.arrivalProgress(player), 1.0);
    EXPECT_TRUE(w.has<PendingTransition>(player));
    w.residency().jobHookForTests = nullptr;
    latch().open();
    ASSERT_TRUE(stepUntil(w, [&] { return !w.has<PendingTransition>(player); }));
    EXPECT_NEAR(w.read<Transform>(player).position().x(), target.x(), 1e-9);
    w.step();
    EXPECT_TRUE(w.extract().instanceOf(player));
    EXPECT_TRUE(w.simulated(player));
    w.setRenderSink(nullptr);
}

TEST(Arrival, SyncCompletesNextStep)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    w.step();
    ASSERT_TRUE(w.requestArrival(player, 0, Vec3(5150, 0, 1400)));
    w.step();
    EXPECT_FALSE(w.has<PendingTransition>(player));
    EXPECT_NEAR(w.read<Transform>(player).position().x(), 5150, 1e-9);
}

TEST(Arrival, TransitionIsArrival)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    // A space with terrain, so the island has jobs (test_small: test_valley's 1 m chunks cost four times as much)
    const SpaceId valley = spaceNamed(w, "test_small");
    const Entity player  = addPlayer(w, Vec3(1150, 0, 1400));
    ASSERT_TRUE(w.completeResidency(30));
    const Entity link = w.createIn(0);
    SpaceLink l;
    l.target  = valley;
    l.arrival = Vec3(1150, 0, 1400);
    w.add<SpaceLink>(link, l);
    w.residency().jobHookForTests = holdChunks({});
    ScopedLatch hold;
    EXPECT_TRUE(w.requestTransition(player, link));
    EXPECT_FALSE(w.requestTransition(player, link));
    w.step(3);
    EXPECT_TRUE(w.has<PendingTransition>(player));
    EXPECT_EQ(w.read<Transform>(player).space(), 0u);
    w.residency().jobHookForTests = nullptr;
    latch().open();
    ASSERT_TRUE(stepUntil(w, [&] { return !w.has<PendingTransition>(player); }));
    EXPECT_EQ(w.read<Transform>(player).space(), valley);
}

TEST(Arrival, NonPlayerArrivalUsesS21Rule)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    addPlayer(w, Vec3(1150, 0, 1400));
    w.completeResidency(30);
    const Entity npc = spawnTicker(w, Vec3(1160, 0, 1400));
    w.step();
    ASSERT_TRUE(w.requestArrival(npc, 0, Vec3(5150, 0, 1400)));
    EXPECT_TRUE(w.simulated(npc));
    w.step();
    EXPECT_FALSE(w.has<PendingTransition>(npc));
    EXPECT_NEAR(w.read<Transform>(npc).position().x(), 5150, 1e-9);
    w.step();
    EXPECT_FALSE(w.simulated(npc));
    // No centre for it
    for (const StreamingCentre& c : w.residency().centres()) EXPECT_NE(c.entity, npc);
}

TEST(Arrival, IslandLimit)
{
    WorldSetup s                  = streamingTestSetup(sharedStreamingCache());
    s.config.maxSimulationIslands = 2;
    World w(s);
    w.setStreamingSynchronous(true);
    addPlayer(w, Vec3(1150, 0, 1400));
    addPlayer(w, Vec3(5150, 0, 1400));
    addPlayer(w, Vec3(9150, 0, 1400));
    std::vector<uint32_t> counts = stepCollectingIslandEvents(w, 5);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts[0], 3u);
    EXPECT_GE(w.residency().stats().islandLimitSteps, 4u);
    EXPECT_EQ(w.islandCount(), 3);
    addPlayer(w, Vec3(15150, 0, 1400));
    counts = stepCollectingIslandEvents(w, 3);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts[0], 4u);
    // The hard limit
    WorldSetup hard             = s;
    hard.config.hardIslandLimit = true;
    World h(hard);
    h.setStreamingSynchronous(true);
    const Entity p1 = addPlayer(h, Vec3(1150, 0, 1400));
    addPlayer(h, Vec3(5150, 0, 1400));
    const Entity p3 = addPlayer(h, Vec3(5200, 0, 1400));
    h.step();
    EXPECT_FALSE(h.requestArrival(p3, 0, Vec3(9150, 0, 1400)));
    EXPECT_TRUE(h.requestArrival(p3, 0, Vec3(1200, 0, 1400)));
    const Entity link = h.createIn(0);
    SpaceLink l;
    l.target  = 0;
    l.arrival = Vec3(12000, 0, 1400);
    h.add<SpaceLink>(link, l);
    // p4 joins p2's island (p3 counts at its accepted target next to p1); its transition would make a third: refused
    const Entity p4 = addPlayer(h, Vec3(5180, 0, 1400));
    EXPECT_FALSE(h.requestTransition(p4, link));
    (void)p1;
}

// Regression (CodeRabbit on #90): two arrivals that each pass the hard limit alone must not be accepted together; the
// accepted arrivals count at their targets
TEST(Arrival, HardLimitCountsAcceptedArrivals)
{
    WorldSetup s                  = streamingTestSetup(sharedStreamingCache());
    s.config.maxSimulationIslands = 2;
    s.config.hardIslandLimit      = true;
    World h(s);
    h.setStreamingSynchronous(true);
    const Entity a = addPlayer(h, Vec3(1150, 0, 1400));
    const Entity b = addPlayer(h, Vec3(1160, 0, 1400));
    addPlayer(h, Vec3(1170, 0, 1400));
    h.step();
    EXPECT_EQ(h.islandCount(), 1);
    EXPECT_TRUE(h.requestArrival(a, 0, Vec3(5150, 0, 1400)));   // 2 islands once it arrives
    EXPECT_FALSE(h.requestArrival(b, 0, Vec3(9150, 0, 1400)));  // a third: refused although a is still at home
}

TEST(Arrival, Refusals)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    EXPECT_FALSE(w.requestArrival(player, SpaceId(9999), Vec3(1, 0, 1)));
    EXPECT_FALSE(w.requestArrival(player, 0, Vec3(std::nan(""), 0, 1)));
    EXPECT_FALSE(w.requestArrival(player, 0, Vec3(-10, 0, 1)));
    EXPECT_FALSE(w.requestArrival(player, 0, Vec3(1e6, 0, 1)));
    EXPECT_TRUE(w.requestArrival(player, 0, Vec3(2000, 0, 2000)));
    EXPECT_FALSE(w.requestArrival(player, 0, Vec3(3000, 0, 2000)));
#ifdef SAIGA_ASSERTS
    auto net      = LoopbackTransport::createNetwork(2, LoopbackOptions());
    WorldSetup cs = streamingTestSetup(sharedStreamingCache());
    cs.transport  = net[1].get();
    cs.role       = NetRole::Client;
    cs.localPeer  = 1;
    World client(cs);
    const Entity e = client.createIn(0);
    EXPECT_DEATH(client.requestArrival(e, 0, Vec3(1, 0, 1)), "requestArrival on a client");
#endif
}

TEST(Arrival, StreamingOffUnchanged)
{
    WorldSetup s       = streamingTestSetup(sharedStreamingCache());
    s.config.streaming = false;
    World w(s);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    w.setRenderMesh(player, "test/chest");
    w.step();
    ASSERT_TRUE(w.requestArrival(player, 0, Vec3(5150, 0, 1400)));
    EXPECT_TRUE(w.simulated(player));
    EXPECT_TRUE(w.extract().instanceOf(player));
    w.step();
    EXPECT_FALSE(w.has<PendingTransition>(player));
    EXPECT_TRUE(w.extract().instanceOf(player));
    w.setRenderSink(nullptr);
}

// ==== StreamingSave =================================================================================================

TEST(StreamingSave, RebuildBeforeLoadReturns)
{
    WorldSetup s = streamingTestSetup(sharedStreamingCache());
    World w(s);
    const Entity player = addPlayer(w, Vec3(1150, 0, 1400));
    w.step();
    ASSERT_TRUE(w.requestArrival(player, 0, Vec3(5150, 0, 1400)));
    const std::vector<uint8_t> saved = w.save();
    World loaded(s);
    ASSERT_TRUE(loaded.tryLoad(saved).ok());
    // The required chunks of the loaded centres (the player and its arrival target) are resident, no step ran
    for (const ChunkId& c : chunksWithin(loaded, 0, Vec3(1150, 0, 1400), loaded.spaceTemplate(0).requiredDistance()))
        EXPECT_TRUE(loaded.residency().ready(c));
    for (const ChunkId& c : chunksWithin(loaded, 0, Vec3(5150, 0, 1400), loaded.spaceTemplate(0).requiredDistance()))
        EXPECT_TRUE(loaded.residency().ready(c));
    EXPECT_EQ(loaded.save(), saved);
    loaded.step();
    EXPECT_FALSE(loaded.has<PendingTransition>(player));
    EXPECT_NEAR(loaded.read<Transform>(player).position().x(), 5150, 1e-9);
}

TEST(StreamingSave, SavedFlagsAreAllSet)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    w.setStreamingSynchronous(true);
    addPlayer(w, Vec3(1150, 0, 1400));
    const Entity far = spawnTicker(w, Vec3(1350, 0, 1400));
    w.step();
    ASSERT_FALSE(w.simulated(far));
    // The ENTS section: u32 slot count, then per slot (from 1) the generation u32, the state u8 and the flags u8
    const std::vector<uint8_t> blob = w.save();
    size_t pos = 8, ents = 0;
    while (pos + 12 <= blob.size())
    {
        uint32_t tag;
        uint64_t len;
        std::memcpy(&tag, blob.data() + pos, 4);
        std::memcpy(&len, blob.data() + pos + 4, 8);
        if (tag == kTagEntities)
        {
            ents = pos + 12;
            break;
        }
        pos += 12 + size_t(len);
    }
    ASSERT_GT(ents, 0u);
    uint32_t slots;
    std::memcpy(&slots, blob.data() + ents, 4);
    int alive = 0;
    for (uint32_t i = 1; i < slots; ++i)
    {
        const size_t at = ents + 4 + size_t(i - 1) * 6;
        if (blob[at + 4] != uint8_t(SlotState::Alive)) continue;
        alive++;
        EXPECT_TRUE(blob[at + 5] & Registry::kSimulated) << "slot " << i;
    }
    EXPECT_EQ(alive, int(w.registry().aliveCount()));
}

TEST(StreamingSave, FlagsContinueExactly)
{
    WorldSetup s = streamingTestSetup(sharedStreamingCache());
    World a(s);
    a.setStreamingSynchronous(true);
    const Entity p = addPlayer(a, Vec3(1150, 0, 1400));
    for (int i = 0; i < 30; ++i) a.spawn<Drifter>(Vec3(1100 + 10.0 * i, 0, 1350 + 5.0 * i));
    a.step(5);
    World b(s);
    b.setStreamingSynchronous(true);
    ASSERT_TRUE(b.tryLoad(a.save()).ok());
    auto same = [&](int step)
    {
        for (uint32_t i = 1; i < a.registry().slotCount(); ++i)
        {
            const Entity e = a.registry().aliveAt(i);
            if (!e.valid()) continue;
            ASSERT_EQ(a.simulated(e), b.simulated(e)) << "step " << step << " entity " << i;
        }
    };
    same(-1);
    for (int k = 0; k < 50; ++k)
    {
        if (k == 10)
        {
            ASSERT_TRUE(a.requestArrival(p, 0, Vec3(5150, 0, 1400)));
            ASSERT_TRUE(b.requestArrival(p, 0, Vec3(5150, 0, 1400)));
        }
        if (k == 30)
        {
            a.get<Transform>(p).teleport(Vec3(1200, 0, 1400), Quat::Identity());
            b.get<Transform>(p).teleport(Vec3(1200, 0, 1400), Quat::Identity());
        }
        a.step();
        b.step();
        same(k);
        if (::testing::Test::HasFatalFailure()) return;
    }
    EXPECT_EQ(a.hash(), b.hash());
}

TEST(StreamingSave, IslandLimitNotRepeated)
{
    WorldSetup s                  = streamingTestSetup(sharedStreamingCache());
    s.config.maxSimulationIslands = 1;
    World w(s);
    w.setStreamingSynchronous(true);
    addPlayer(w, Vec3(1150, 0, 1400));
    addPlayer(w, Vec3(5150, 0, 1400));
    w.step(3);
    EXPECT_EQ(w.readGlobal<IslandState>().reportedCount, 2u);
    World loaded(s);
    loaded.setStreamingSynchronous(true);
    ASSERT_TRUE(loaded.tryLoad(w.save()).ok());
    EXPECT_TRUE(stepCollectingIslandEvents(loaded, 4).empty());
    EXPECT_EQ(loaded.readGlobal<IslandState>().reportedCount, 2u);
}

TEST(StreamingSave, StreamingOffSetsEveryFlag)
{
    WorldSetup s       = streamingTestSetup(sharedStreamingCache());
    s.config.streaming = false;
    World w(s);
    const Entity e = spawnTicker(w, Vec3(1, 0, 1));
    w.registry().setFlags(e, 0);
    World loaded(s);
    ASSERT_TRUE(loaded.tryLoad(w.save()).ok());
    EXPECT_TRUE(loaded.simulated(e));
}

TEST(StreamingSave, BetweenStepNpcChangeMatchesLoad)
{
    WorldSetup s = streamingTestSetup(sharedStreamingCache());
    World w(s);
    w.setStreamingSynchronous(true);
    addPlayer(w, Vec3(1150, 0, 1400));
    const Entity farRoot = spawnTicker(w, Vec3(1350, 0, 1400));
    const Entity npcA    = spawnTicker(w, Vec3(1360, 0, 1400));
    const Entity npcB    = spawnTicker(w, Vec3(1160, 0, 1400));
    const Entity npcC    = spawnTicker(w, Vec3(1370, 0, 1400));
    w.step(2);
    // Between steps: a teleport next to the player, a re-parent under the far root, an alwaysSimulate edit
    w.get<Transform>(npcA).teleport(Vec3(1155, 0, 1400), Quat::Identity());
    w.add<Parent>(npcB, Parent{farRoot});
    w.get<Behaviour>(npcC).alwaysSimulate = true;
    // A plain component write is not a structural change: the writer re-evaluates (the world API does it for
    // set_component / add_component / remove_component of Behaviour)
    w.refreshSimulated(npcC);
    World loaded(s);
    loaded.setStreamingSynchronous(true);
    ASSERT_TRUE(loaded.tryLoad(w.save()).ok());
    // The plain world's flags in the next phase 2 (after its step start) equal the loaded copy's
    w.step();
    loaded.step();
    for (Entity e : {farRoot, npcA, npcB, npcC}) EXPECT_EQ(w.simulated(e), loaded.simulated(e)) << e.index;
    EXPECT_TRUE(w.simulated(npcA));
    EXPECT_FALSE(w.simulated(npcB));
    EXPECT_TRUE(w.simulated(npcC));
    EXPECT_GT(updates(w, npcA), 0u);
}

// ==== StreamingAbort ================================================================================================

TEST(StreamingAbort, Phase7Abort)
{
    WorldSetup s                  = streamingTestSetup(sharedStreamingCache());
    s.systems                     = abortSystems;
    s.config.maxSimulationIslands = 1;
    World w(s);
    w.setStreamingSynchronous(true);
    addPlayer(w, Vec3(1150, 0, 1400));
    addPlayer(w, Vec3(5150, 0, 1400));
    g_throwPhase7 = true;
    EXPECT_THROW(w.step(), std::runtime_error);
    g_throwPhase7 = false;
    // The residency's results stay applied; no event and no IslandState change were committed
    EXPECT_TRUE(w.chunkReady(0, 1150, 1400));
    EXPECT_EQ(w.readGlobal<IslandState>().reportedCount, 0u);
    // The next step continues and emits the event
    const std::vector<uint32_t> counts = stepCollectingIslandEvents(w, 3);
    EXPECT_EQ(w.readGlobal<IslandState>().reportedCount, 2u);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts[0], 2u);
}

TEST(StreamingAbort, Phase8Abort)
{
    WorldSetup s                  = streamingTestSetup(sharedStreamingCache());
    s.systems                     = abortSystems;
    s.config.maxSimulationIslands = 1;
    World w(s);
    w.setStreamingSynchronous(true);
    addPlayer(w, Vec3(1150, 0, 1400));
    addPlayer(w, Vec3(5150, 0, 1400));
    g_throwPhase8 = true;
    EXPECT_THROW(w.step(), std::runtime_error);
    g_throwPhase8 = false;
    // The state was written at the phase-7 flush; the event is lost (G14, documented)
    EXPECT_EQ(w.readGlobal<IslandState>().reportedCount, 2u);
    EXPECT_TRUE(w.chunkReady(0, 1150, 1400));
    EXPECT_TRUE(stepCollectingIslandEvents(w, 3).empty());
}

TEST(StreamingAbort, HeldJobAcrossLoad)
{
    World w(streamingTestSetup(sharedStreamingCache()));
    const ChunkId c               = chunkOf(0, 1150, 1400);
    w.residency().jobHookForTests = holdChunks({c});
    addPlayer(w, Vec3(1150, 0, 1400));
    ScopedLatch hold;
    w.step();
    const std::vector<uint8_t> saved = w.save();
    w.residency().jobHookForTests    = nullptr;
    // A load with the same pins: the held job's completion delivers nothing (another generation)
    ASSERT_TRUE(w.tryLoad(saved).ok());
    latch().open();
    ASSERT_TRUE(stepUntil(w, [&] { return w.residency().stats().pendingJobs == 0 && w.residency().ready(c); }));
    EXPECT_EQ(w.residency().stats().pendingJobs, 0);
}

// ==== StreamingNet ==================================================================================================

namespace
{
struct NetPair
{
    std::vector<std::unique_ptr<LoopbackTransport>> net;
    std::unique_ptr<World> host, client;
    explicit NetPair(int delay = 0)
    {
        LoopbackOptions o;
        o.delayRounds = delay;
        net           = LoopbackTransport::createNetwork(2, o);
        WorldSetup hs = streamingTestSetup(sharedStreamingCache());
        hs.transport  = net[0].get();
        host          = std::make_unique<World>(hs);
        WorldSetup cs = hs;
        cs.transport  = net[1].get();
        cs.role       = NetRole::Client;
        cs.localPeer  = 1;
        client        = std::make_unique<World>(cs);
        host->setStreamingSynchronous(true);
        client->setStreamingSynchronous(true);
    }
    void step(int n = 1)
    {
        for (int i = 0; i < n; ++i)
        {
            host->step();
            client->step();
        }
    }
};
}  // namespace

TEST(StreamingNet, ClientStreamsAroundItsOwnPlayer)
{
    NetPair p;
    const Entity hostPlayer          = addPlayer(*p.host, Vec3(1150, 0, 1400));
    const Entity remote              = addPlayer(*p.host, Vec3(5150, 0, 1400));
    p.host->get<Player>(remote).peer = 1;
    p.host->setOwner(remote, 1);
    const Entity far  = spawnTicker(*p.host, Vec3(20000, 0, 20000));
    const uint64_t g0 = p.client->generation().cache().stats().baseGenerations;
    p.step(6);
    const Entity mine = p.client->replication().entityOfNetId(p.host->replication().netIdOf(remote));
    ASSERT_TRUE(mine.valid());
    // The host requests around both players, the client around its own only
    EXPECT_TRUE(p.host->chunkReady(0, 1150, 1400));
    EXPECT_TRUE(p.host->chunkReady(0, 5150, 1400));
    EXPECT_TRUE(p.client->chunkReady(0, 5150, 1400));
    EXPECT_FALSE(p.client->residency().requested(chunkOf(0, 1150, 1400)));
    EXPECT_GE(p.client->generation().cache().stats().baseGenerations, g0);
    // Replicated far entities are simulated on the client (only the held ones are not)
    const Entity farOnClient = p.client->replication().entityOfNetId(p.host->replication().netIdOf(far));
    ASSERT_TRUE(farOnClient.valid());
    EXPECT_TRUE(p.client->simulated(farOnClient));
    EXPECT_FALSE(p.host->simulated(far));
    // A host arrival of the client's player (the host's island held on the latch): PendingTransition reaches the
    // client, which holds its player and preloads the target before the host moves it
    p.host->setStreamingSynchronous(false);
    p.host->residency().jobHookForTests = holdChunks({});
    ScopedLatch hold;
    ASSERT_TRUE(p.host->requestArrival(remote, 0, Vec3(9150, 0, 1400)));
    const Vec3 hostPose = p.host->read<Transform>(remote).position();
    p.step(3);
    EXPECT_TRUE(p.client->has<PendingTransition>(mine));
    EXPECT_FALSE(p.client->simulated(mine));
    EXPECT_TRUE(p.client->residency().requested(chunkOf(0, 9150, 1400)));
    EXPECT_TRUE(p.client->chunkReady(0, 9150, 1400));
    // The client's mover holds its player; the host keeps the held root's pose (owner states dropped)
    p.client->get<CharacterIntent>(mine).desiredVelocity = Vec3(5, 0, 0);
    p.step(3);
    EXPECT_EQ(p.host->read<Transform>(remote).position(), hostPose);
    p.host->residency().jobHookForTests = nullptr;
    latch().open();
    const double end = seconds() + 20;
    while (p.client->has<PendingTransition>(mine) || p.client->residency().clientAwaitsMove(mine))
    {
        ASSERT_LT(seconds(), end);
        p.host->terrain().streamingPool().waitIdleFor(0.002);
        p.step();
    }
    p.step();
    EXPECT_NEAR(p.client->read<Transform>(mine).position().x(), 9150, 1.0);
    EXPECT_TRUE(p.client->simulated(mine));
    (void)hostPlayer;
}

// The client keeps an arrival's target as a centre until the move's OwnerCommand is applied (the component's removal
// on the State channel can overtake the command on the Control channel)
TEST(StreamingNet, ClientKeepsTargetUntilMoveApplied)
{
    NetPair p(2);
    const Entity remote              = addPlayer(*p.host, Vec3(5150, 0, 1400));
    p.host->get<Player>(remote).peer = 1;
    p.host->setOwner(remote, 1);
    p.step(8);
    const Entity mine = p.client->replication().entityOfNetId(p.host->replication().netIdOf(remote));
    ASSERT_TRUE(mine.valid());
    // The host's island held for a few steps (the arrival is pending and replicated meanwhile)
    p.host->setStreamingSynchronous(false);
    p.host->residency().jobHookForTests = holdChunks({});
    ScopedLatch hold;
    ASSERT_TRUE(p.host->requestArrival(remote, 0, Vec3(9150, 0, 1400)));
    p.step(4);
    p.host->residency().jobHookForTests = nullptr;
    latch().open();
    // Until the client stands at the target: whenever its player is not there yet, the target is a centre and the
    // player is held (no unfreeze at the old position)
    bool sawPending = false;
    for (int i = 0; i < 40; ++i)
    {
        p.host->terrain().streamingPool().waitIdleFor(0.002);
        p.step();
        const bool there = std::abs(p.client->read<Transform>(mine).position().x() - 9150) < 1e-6;
        if (p.client->has<PendingTransition>(mine)) sawPending = true;
        if (sawPending && !there)
        {
            EXPECT_TRUE(p.client->residency().requested(chunkOf(0, 9150, 1400))) << i;
            EXPECT_FALSE(p.client->simulated(mine)) << i;
        }
    }
    EXPECT_TRUE(sawPending);
    EXPECT_TRUE(p.client->simulated(mine));
    EXPECT_FALSE(p.client->residency().clientAwaitsMove(mine));
}
