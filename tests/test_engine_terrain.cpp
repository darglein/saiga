/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine's terrain without a Vulkan device (docs/engine_design.md §12.2, §12.3; S20): spaces and their validation,
// World::terrainHeight against the integer surface rule, the kinematic mover on the terrain.

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/core/StreamingPool.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/terrain/TerrainGeneration.h"
#include "saiga/engine/terrain/TerrainStore.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/World.h"
#include "saiga/vulkan/scene/TerrainAtlas.h"
#include "saiga/vulkan/scene/TerrainTiles.h"
#include "saiga/vulkan/world/PrototypeTerrain.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

#include "terrain_test_hooks.h"
#include <condition_variable>

using namespace Saiga;
using namespace Saiga::Engine;
using namespace TerrainTestHooks;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

// Heights with negative values, steep and flat parts (the rule test)
int32_t ruleHook(uint64_t seed, int64_t x, int64_t z)
{
    return int32_t(highFrequencyHook(seed, x, z) - 23 * 1024 + triangleWave(x * 3 + z, 2000) * 900 -
                   triangleWave(z, 600) * 700);
}

WorldSetup terrainSetup(int32_t sizeChunks, int baseSpacing, TerrainHeightHook hook,
                        std::optional<int32_t> seaLevel = std::nullopt, uint64_t seed = 5)
{
    WorldSetup s;
    s.seed      = seed;
    s.profiling = false;
    s.config.spaceTemplates.push_back(testSpaceTemplate(sizeChunks, baseSpacing, hook, seaLevel));
    return s;
}
}  // namespace

// §22 acceptance (S20): terrainHeight is the integer rule over the clamped hook samples on the (0,0)-(1,1) diagonal,
// for points on the 1/256 m grid: on and next to diagonals, edges and vertices, near 30 km, below the sea and on the
// far edges of the space
TEST(Terrain, TerrainHeightEqualsTheIntegerRule)
{
    for (int spacing : {1, 2})
    {
        const int32_t chunks = 128;
        World world(terrainSetup(chunks, spacing, ruleHook));
        const uint64_t seed   = world.spaceSeed(0);
        const int f           = spacing == 1 ? 8 : 9;         // a 1/256 m point is a multiple of 2^-f cells
        const int64_t perCell = int64_t(256) * spacing;       // 1/256 m units per cell
        const int64_t size    = int64_t(chunks) * 256 * 256;  // the space in 1/256 m units
        const int64_t cells   = int64_t(chunks) * 256 / spacing;
        std::mt19937_64 rng(uint64_t(11 + spacing));
        auto sample = [&](int64_t i, int64_t j)
        { return clampTerrainHeight(ruleHook(seed, i * spacing, j * spacing)); };
        for (int k = 0; k < 10000; ++k)
        {
            int64_t X = int64_t(rng() % uint64_t(size + 1)), Z = int64_t(rng() % uint64_t(size + 1));
            switch (k % 8)
            {
                case 0:  // on a diagonal
                    Z = (Z / perCell) * perCell + X % perCell;
                    break;
                case 1:  // next to a diagonal
                    Z = std::min(size, (Z / perCell) * perCell + X % perCell + (k % 16 == 1 ? 1 : -1));
                    break;
                case 2:  // a cell edge
                    X = (X / perCell) * perCell;
                    break;
                case 3:  // a vertex
                    X = (X / perCell) * perCell;
                    Z = (Z / perCell) * perCell;
                    break;
                case 4:  // near 30 km
                    X = int64_t(30000) * 256 + int64_t(rng() % (256 * 600));
                    Z = int64_t(29000) * 256 + int64_t(rng() % (256 * 600));
                    break;
                case 5:  // the far edges: the last cell with fx = 1
                    if (k % 16 == 5)
                        X = size;
                    else
                        Z = size;
                    break;
                default:
                    break;
            }
            X                = std::max<int64_t>(0, std::min(size, X));
            Z                = std::max<int64_t>(0, std::min(size, Z));
            const int64_t ci = std::min(X / perCell, cells - 1), cj = std::min(Z / perCell, cells - 1);
            const int64_t fx = X - ci * perCell, fz = Z - cj * perCell;
            const int32_t h00 = sample(ci, cj), h10 = sample(ci + 1, cj), h01 = sample(ci, cj + 1),
                          h11 = sample(ci + 1, cj + 1);
            const double h    = world.terrainHeight(0, double(X) / 256, double(Z) / 256);
            const double hu   = h / kHeightUnit;
            ASSERT_EQ(int64_t(std::floor(hu)), int64_t(terrainSurfaceUnits(h00, h10, h01, h11, fx, fz, f)))
                << "spacing " << spacing << " at " << X << ", " << Z;
            ASSERT_EQ(hu * double(int64_t(1) << f), double(terrainSurfaceNumerator(h00, h10, h01, h11, fx, fz, f)));
            if (fx % (int64_t(1) << f) == 0 && fz % (int64_t(1) << f) == 0)
            {
                ASSERT_EQ(hu, double(sample(ci + fx / (int64_t(1) << f), cj + fz / (int64_t(1) << f))));
            }
        }
        // Outside the space: clamped to it
        EXPECT_EQ(world.terrainHeight(0, -100, -5), world.terrainHeight(0, 0, 0));
        EXPECT_EQ(world.terrainHeight(0, 1e9, 17.5), world.terrainHeight(0, chunks * 256.0, 17.5));
    }
}

// prototypeSurfaceHeight equals World::terrainHeight bit for bit for the prototype's template (built inline from the
// PrototypeTerrain constants): the flight and racing contexts rely on it
TEST(Terrain, PrototypeSurfaceEqualsTerrainHeight)
{
    for (int bs : {1, 2})
    {
        WorldSetup s;
        s.seed      = 3;
        s.profiling = false;
        s.config.spaceTemplates.push_back(
            testSpaceTemplate(Vulkan::kPrototypeSpaceChunks, bs, Vulkan::prototypeHeight, Vulkan::kPrototypeSeaLevel));
        World w(s);
        std::mt19937_64 rng{uint64_t(bs)};
        for (int k = 0; k < 1000; ++k)
        {
            double x = double(rng() % (32768 * 64)) / 64, z = double(rng() % (32768 * 64)) / 64;
            if (k % 50 == 0) x = 32768;
            if (k % 50 == 1) z = 0;
            if (k % 50 == 2) x = -3;
            ASSERT_EQ(Vulkan::prototypeSurfaceHeight(3, bs, x, z), w.terrainHeight(0, x, z)) << x << ", " << z;
        }
        // Regression (CodeRabbit, S20): a NaN coordinate survived the clamps and reached int64_t(floor(NaN)) (UB); 0
        // as terrainHeight
        const double nan = std::numeric_limits<double>::quiet_NaN();
        EXPECT_EQ(Vulkan::prototypeSurfaceHeight(3, bs, nan, 100), w.terrainHeight(0, nan, 100));
        EXPECT_EQ(Vulkan::prototypeSurfaceHeight(3, bs, 100, nan), 0.0);
    }
}

TEST(Terrain, FlatWithoutHook)
{
    WorldSetup setup;
    setup.profiling = false;
    World world(setup);
    ASSERT_EQ(world.spaceCount(), 1u);
    EXPECT_FALSE(world.spaceTemplate(0).hasTerrain());
    std::mt19937_64 rng(3);
    for (int k = 0; k < 1000; ++k)
    {
        const int64_t x = int64_t(rng() % 2000000) - 1000000, z = int64_t(rng() % 2000000) - 1000000;
        EXPECT_EQ(world.terrainHeight(0, double(x) * 0.37, double(z) * 0.11), 0.0);
        EXPECT_EQ(world.terrain().sample(0, x, z), 0);
    }
}

// Every hook result passes clampTerrainHeight (fromHook)
TEST(Terrain, HookResultsClamped)
{
    World world(
        terrainSetup(16, 1, [](uint64_t, int64_t x, int64_t)
                     { return x % 2 ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int32_t>::min(); }));
    EXPECT_EQ(world.terrain().sample(0, 1, 0), kTerrainHeightLimit);
    EXPECT_EQ(world.terrain().sample(0, 2, 0), -kTerrainHeightLimit);
    EXPECT_EQ(world.terrainHeight(0, 1, 0), kTerrainHeightLimit * kHeightUnit);
    EXPECT_EQ(world.terrainHeight(0, 0.5, 0), 0.0);
}

// Regression: a NaN coordinate survived the clamps of TerrainSystem::height and reached int64_t(floor(NaN)) (UB); it
// is 0 now, infinities clamp to the edges
TEST(Terrain, NonFiniteCoordinates)
{
    World world(terrainSetup(16, 1, slopeRidgeHook));
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    EXPECT_EQ(world.terrainHeight(0, nan, 100), 0.0);
    EXPECT_EQ(world.terrainHeight(0, 100, nan), 0.0);
    EXPECT_EQ(world.terrainHeight(0, inf, 100), world.terrainHeight(0, 16 * 256, 100));
    EXPECT_EQ(world.terrainHeight(0, 100, -inf), world.terrainHeight(0, 100, 0));
}

TEST(Terrain, SpaceSeeds)
{
    World world(terrainSetup(16, 2, slopeRidgeHook, 0, 77));
    EXPECT_EQ(world.spaceSeed(0), 77u);
    // The space's hook gets the space seed
    EXPECT_EQ(world.terrain().sample(0, 3, 4), slopeRidgeHook(77, 6, 8));
}

TEST(TerrainDeathTest, TemplateValidation)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    auto with = [](std::vector<SpaceTemplate> list)
    {
        WorldSetup s;
        s.profiling             = false;
        s.config.spaceTemplates = std::move(list);
        World w(s);
    };
    SpaceTemplate ok;
    for (int32_t chunks : {8, 520, 100})
    {
        SpaceTemplate t = ok;
        t.sizeChunks    = chunks;
        EXPECT_DEATH(with({t}), "sizeChunks");
    }
    SpaceTemplate spacing = ok;
    spacing.baseSpacing   = 3;
    EXPECT_DEATH(with({spacing}), "baseSpacing");
    SpaceTemplate sea = ok;
    sea.seaLevel      = kTerrainHeightLimit + 1;
    EXPECT_DEATH(with({sea}), "seaLevel");
    // Valid: the bounds
    SpaceTemplate a = ok, b = ok;
    a.sizeChunks  = 16;
    b.sizeChunks  = 512;
    b.baseSpacing = 1;
    with({a});
    with({b});
}

// ---- the kinematic mover (characterMoveSystem, §8.2, §9.3)

namespace
{
Entity character(World& w, const Vec3& position, const Vec3& velocity)
{
    Entity e = w.create();
    w.get<Transform>(e).teleport(position, Quat::Identity());
    CharacterIntent intent;
    intent.desiredVelocity = velocity;
    w.add<CharacterIntent>(e, intent);
    return e;
}
}  // namespace

TEST(CharacterMove, FollowsTerrain)
{
    World w(terrainSetup(16, 1, slopeRidgeHook));
    Entity e = character(w, Vec3(100, 0, 100), Vec3(3, 0, 1));
    for (int i = 0; i < 300; ++i)
    {
        w.step();
        const Vec3 p = w.read<Transform>(e).position();
        ASSERT_EQ(p.y(), w.terrainHeight(0, p.x(), p.z())) << "step " << i;
    }
    EXPECT_GT(w.read<Transform>(e).position().x(), 110);
}

// In a terrain space a teleported idle character is snapped in the next step; a resting one is not written
TEST(CharacterMove, IdleSnapsOnlyWhenOff)
{
    World w(terrainSetup(16, 2, slopeRidgeHook));
    Entity e = character(w, Vec3(50.25, 400, 70.5), Vec3::Zero());
    w.step();
    const double ground = w.terrainHeight(0, 50.25, 70.5);
    EXPECT_EQ(w.read<Transform>(e).position().y(), ground);
    EXPECT_EQ(w.read<Transform>(e).position().x(), 50.25);
    w.step(2);
    EXPECT_FALSE(w.read<Transform>(e).changedInThisStep());
    EXPECT_FALSE(w.read<Transform>(e).changedInLastStep());
    // Teleported again: snapped again
    w.get<Transform>(e).teleport(Vec3(10, -30, 12), Quat::Identity());
    w.step();
    EXPECT_EQ(w.read<Transform>(e).position().y(), w.terrainHeight(0, 10, 12));
}

// The flat default: an idle character at y = 3 stays there (as before S20)
TEST(CharacterMove, IdleUnchangedWithoutTerrain)
{
    WorldSetup s;
    s.profiling = false;
    World w(s);
    Entity e = character(w, Vec3(1, 3, 2), Vec3::Zero());
    w.step(5);
    EXPECT_EQ(w.read<Transform>(e).position(), Vec3(1, 3, 2));
    // A moving one is at y = 0
    w.get<CharacterIntent>(e).desiredVelocity = Vec3(1, 0, 0);
    w.step();
    EXPECT_EQ(w.read<Transform>(e).position().y(), 0.0);
}

// ---- StreamingPool (§8.5, D8)

namespace
{
// A gate that held jobs wait on
struct Gate
{
    std::mutex m;
    std::condition_variable cv;
    bool open = false;
    std::atomic<int> waiting{0};
    void wait()
    {
        std::unique_lock<std::mutex> lock(m);
        waiting++;
        cv.wait(lock, [this]() { return open; });
        waiting--;
    }
    void release()
    {
        {
            std::lock_guard<std::mutex> lock(m);
            open = true;
        }
        cv.notify_all();
    }
    // Until n jobs wait (or 5 s)
    bool awaitWaiting(int n)
    {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (waiting.load() < n && std::chrono::steady_clock::now() < end)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return waiting.load() >= n;
    }
};
// Declared after the gate and the state the jobs capture: releases the gate and waits for the pool before they are
// destroyed, also when a fatal assertion returns early (regression: a failed ASSERT left a worker waiting on a
// destroyed gate)
struct GateRelease
{
    Gate& gate;
    StreamingPool& pool;
    ~GateRelease()
    {
        gate.release();
        pool.waitIdle();
    }
};
}  // namespace

TEST(StreamingPool, PriorityOrderAndFifo)
{
    StreamingPool pool(1);
    Gate gate;
    std::vector<int> order;
    std::mutex orderMutex;
    GateRelease guard{gate, pool};
    // The only worker is held, so the rest queue up
    pool.submit([&](const std::atomic<bool>&) { gate.wait(); }, 0, nullptr);
    ASSERT_TRUE(gate.awaitWaiting(1));
    const std::vector<std::pair<int, int>> jobs = {{5, 0}, {1, 1}, {5, 2}, {0, 3}, {1, 4}};
    for (auto [priority, tag] : jobs)
        pool.submit(
            [&, tag = tag](const std::atomic<bool>&)
            {
                std::lock_guard<std::mutex> lock(orderMutex);
                order.push_back(tag);
            },
            priority, nullptr);
    EXPECT_EQ(pool.queued(), 5);
    gate.release();
    pool.waitIdle();
    EXPECT_EQ(order, (std::vector<int>{3, 1, 4, 0, 2}));
    // The completions run in finishing order at drain
    std::vector<int> completions;
    pool.submit([](const std::atomic<bool>&) {}, 0, [&](bool c) { completions.push_back(c ? -1 : 1); });
    pool.waitIdle();
    EXPECT_TRUE(completions.empty());
    pool.drain();
    EXPECT_EQ(completions, std::vector<int>{1});
}

TEST(StreamingPool, CancelQueuedAndRunning)
{
    StreamingPool pool(1);
    Gate gate;
    std::atomic<bool> sawCancel{false};
    int runningDone = 0, queuedDone = 0;
    GateRelease guard{gate, pool};
    const uint64_t running = pool.submit(
        [&](const std::atomic<bool>& cancelled)
        {
            gate.wait();
            sawCancel = cancelled.load();
        },
        0,
        [&](bool cancelled)
        {
            EXPECT_TRUE(cancelled);
            runningDone++;
        });
    ASSERT_TRUE(gate.awaitWaiting(1));
    const uint64_t queued =
        pool.submit([](const std::atomic<bool>&) { FAIL() << "a cancelled job ran"; }, 0, [&](bool) { queuedDone++; });
    EXPECT_TRUE(pool.cancel(queued));
    EXPECT_EQ(pool.queued(), 0);
    EXPECT_FALSE(pool.cancel(running));
    EXPECT_FALSE(pool.cancel(12345));
    gate.release();
    pool.waitIdle();
    pool.drain();
    EXPECT_TRUE(sawCancel.load());
    EXPECT_EQ(runningDone, 1);
    EXPECT_EQ(queuedDone, 0);
}

TEST(StreamingPool, CountsWithHeldJobs)
{
    StreamingPool pool(2);
    Gate gate;
    std::vector<uint64_t> ids;
    GateRelease guard{gate, pool};
    for (int i = 0; i < 5; ++i) ids.push_back(pool.submit([&](const std::atomic<bool>&) { gate.wait(); }, 0, nullptr));
    ASSERT_TRUE(gate.awaitWaiting(2));
    EXPECT_EQ(pool.running(), 2);
    EXPECT_EQ(pool.queued(), 3);
    EXPECT_EQ(pool.finished(), 0);
    // A queued cancel drops the count at once
    EXPECT_TRUE(pool.cancel(ids[4]));
    EXPECT_EQ(pool.queued(), 2);
    // waitIdleFor returns false while a held job runs
    EXPECT_FALSE(pool.waitIdleFor(0.05));
    gate.release();
    EXPECT_TRUE(pool.waitIdleFor(5));
    EXPECT_EQ(pool.running(), 0);
    EXPECT_EQ(pool.queued(), 0);
    EXPECT_EQ(pool.finished(), 4);
    pool.drain();
    EXPECT_EQ(pool.finished(), 0);
}

TEST(StreamingPool, RunOneOnCaller)
{
    StreamingPool pool(1);
    Gate gate;
    GateRelease guard{gate, pool};
    pool.submit([&](const std::atomic<bool>&) { gate.wait(); }, 0, nullptr);
    ASSERT_TRUE(gate.awaitWaiting(1));
    std::thread::id ranOn;
    int done = 0;
    pool.submit([](const std::atomic<bool>&) {}, 9, nullptr);
    pool.submit([&](const std::atomic<bool>&) { ranOn = std::this_thread::get_id(); }, 2, [&](bool) { done++; });
    // The first queued job in priority order runs on this thread; its completion is queued for drain
    EXPECT_TRUE(pool.runOneOnCaller());
    EXPECT_EQ(ranOn, std::this_thread::get_id());
    EXPECT_EQ(done, 0);
    EXPECT_EQ(pool.finished(), 1);
    pool.drain();
    EXPECT_EQ(done, 1);
    EXPECT_TRUE(pool.runOneOnCaller());
    EXPECT_FALSE(pool.runOneOnCaller());
    gate.release();
    pool.waitIdle();
}

TEST(StreamingPool, ExceptionsAtDrain)
{
    StreamingPool pool(1);
    Gate gate;
    GateRelease guard{gate, pool};
    pool.submit([&](const std::atomic<bool>&) { gate.wait(); }, 0, nullptr);
    ASSERT_TRUE(gate.awaitWaiting(1));
    std::vector<int> completions;
    pool.submit([](const std::atomic<bool>&) {}, 0, [&](bool) { completions.push_back(1); });
    pool.submit([](const std::atomic<bool>&) { throw std::runtime_error("generation failed"); }, 0,
                [&](bool) { completions.push_back(99); });
    pool.submit([](const std::atomic<bool>&) {}, 0, [&](bool) { completions.push_back(3); });
    gate.release();
    // waitIdle never throws
    EXPECT_NO_THROW(pool.waitIdle());
    EXPECT_EQ(pool.running(), 0);
    EXPECT_EQ(pool.queued(), 0);
    try
    {
        pool.drain();
        FAIL() << "drain did not rethrow";
    }
    catch (const std::runtime_error& e)
    {
        EXPECT_STREQ(e.what(), "generation failed");
    }
    // The completion before the failed job ran, the one behind it stays queued for the next drain
    EXPECT_EQ(completions, std::vector<int>{1});
    EXPECT_EQ(pool.finished(), 1);
    pool.drain();
    EXPECT_EQ(completions, (std::vector<int>{1, 3}));
}

TEST(StreamingPool, DestructionWithQueuedAndRunningJobs)
{
    std::atomic<int> stopped{0};
    int completions = 0;
    {
        StreamingPool pool(2);
        for (int i = 0; i < 6; ++i)
            pool.submit(
                [&](const std::atomic<bool>& cancelled)
                {
                    while (!cancelled.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    stopped++;
                },
                0, [&](bool) { completions++; });
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (pool.running() < 2 && std::chrono::steady_clock::now() < end)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        EXPECT_EQ(pool.running(), 2);
    }
    // The running jobs saw the cancel; the queued ones never ran; no completion is called
    EXPECT_EQ(stopped.load(), 2);
    EXPECT_EQ(completions, 0);
}

// The latched hook blocks only off the arming thread
TEST(StreamingPool, LatchedHook)
{
    // The pool first: the latch opens before the pool joins its worker
    StreamingPool pool(1);
    std::optional<ScopedLatch> latch;
    latch.emplace();
    std::atomic<int> value{0};
    pool.submit([&](const std::atomic<bool>&) { value = latchedHook(1, 2, 3); }, 0, nullptr);
    EXPECT_EQ(latchedHook(1, 2, 3), integerTestHook(1, 2, 3));
    EXPECT_FALSE(pool.waitIdleFor(0.05));
    EXPECT_EQ(value.load(), 0);
    latch.reset();
    pool.waitIdle();
    EXPECT_EQ(value.load(), integerTestHook(1, 2, 3));
}

// ---------------------------------------------------------------------------------------------------------------------
// Tiles, pyramid, grass masks (the pure generation through TerrainGeneration)

TEST(Terrain, TilesShareBordersAndSubsample)
{
    for (int bs : {1, 2})
    {
        const auto layout = testLayout(16, bs);
        const auto src    = heightSource(testSpaceTemplate(16, bs, highFrequencyHook), 3);
        const int n       = layout.samplesPerSide();
        auto tile         = [&](int level, int x, int z)
        {
            RegionId r;
            r.level = uint8_t(level);
            r.x     = x;
            r.z     = z;
            return *generateTile(src, layout, r).heights;
        };
        for (int level = 0; level + 1 < layout.streamedLevels(); ++level)
        {
            const auto a = tile(level, 2, 3), right = tile(level, 3, 3), below = tile(level, 2, 4);
            for (int j = 0; j < n; ++j) EXPECT_EQ(a[size_t(j * n + n - 1)], right[size_t(j * n)]);
            for (int i = 0; i < n; ++i) EXPECT_EQ(a[size_t((n - 1) * n + i)], below[size_t(i)]);
            // A level-(L+1) sample equals the level-L sample at the same point (point subsamples)
            const auto coarse = tile(level + 1, 1, 1);
            const auto fine   = tile(level, 2, 3);  // the lower-left quarter's upper part of coarse (1, 1)
            for (int j = 0; j < n; j += 2)
                for (int i = 0; i < n; i += 2)
                    EXPECT_EQ(fine[size_t(j * n + i)], coarse[size_t((j / 2 + (n - 1) / 2) * n + i / 2)])
                        << "level " << level;
        }
    }
}

TEST(Terrain, Pyramid)
{
    const auto layout = testLayout(16, 1);
    const auto p      = generatePyramid(heightSource(testSpaceTemplate(16, 1, slopeRidgeHook), 4), layout, 1234);
    const int P       = layout.pyramidSamplesPerSide();
    ASSERT_EQ(P, 129);
    EXPECT_EQ(p.seaLevel, std::optional<int32_t>(1234));
    for (int j = 0; j < P; j += 7)
        for (int i = 0; i < P; i += 5) EXPECT_EQ((*p.heights)[size_t(j * P + i)], slopeRidgeHook(4, i * 32, j * 32));
    int32_t lo = 0, hi = 0;
    Vulkan::terrainHeightRange(*p.heights, lo, hi);
    EXPECT_EQ(p.minHeight, lo);
    EXPECT_EQ(p.maxHeight, hi);
}

TEST(Terrain, GrassMaskRule)
{
    const auto layout = testLayout(16, 2);
    // A slope of h(x) = x * s (height units per metre): normal.y = 1 / sqrt(1 + (s / 1024)^2)
    auto slope = [](uint64_t s, int64_t x, int64_t) { return int32_t(int64_t(s) * x); };
    RegionId r;
    r.x = 1;
    r.z = 1;
    // normal.y 0.82 <=> slope 0.698: 714 units per metre
    for (auto [s, expected] : std::vector<std::pair<uint64_t, int>>{{700, 255}, {730, 0}})
    {
        const auto tile = generateTile(TerrainHeightSource::fromHook(slope, s), layout, r);
        const auto mask = generateGrassMask(tile, layout, std::nullopt, 0.82, 0.4);
        ASSERT_EQ(mask.size, 128);
        for (uint8_t v : *mask.density) ASSERT_EQ(int(v), expected) << "slope " << s;
    }
    // The sea: cells whose lowest sample is below sea + 0.4 m have no grass
    auto flat5      = [](uint64_t, int64_t x, int64_t) { return int32_t(x < 300 ? 5 * 1024 : 7 * 1024); };
    const auto tile = generateTile(TerrainHeightSource::fromHook(flat5, 1), layout, r);
    const auto mask = generateGrassMask(tile, layout, 6 * 1024, 0.82, 0.4);
    // Cells from x = 256: at x < 300 the lowest sample is 5 m (wet), from x = 300 7 m (dry); the cell at 298 has a 300
    // sample but its lowest is 5 m (and a steep step)
    EXPECT_EQ(int((*mask.density)[0]), 0);
    EXPECT_EQ(int((*mask.density)[size_t(127)]), 255);
    const auto noSea = generateGrassMask(tile, layout, std::nullopt, 0.82, 0.4);
    EXPECT_EQ(int((*noSea.density)[0]), 255);
}

// ---------------------------------------------------------------------------------------------------------------------
// Streaming (TerrainSystem with a RecordingRenderSink)

namespace
{
const TerrainViewSettings kDefaults;

RegionId tileRegion(int level, int32_t x, int32_t z)
{
    RegionId r;
    r.level = uint8_t(level);
    r.x     = x;
    r.z     = z;
    return r;
}

// The desired tiles of the default rule (the CPU reference of §4.3, terrain_test_hooks' desiredTestTiles) by level
std::map<int, std::set<RegionId>> referenceDesired(const World& w, const Vec3& camera, const TerrainViewSettings& s)
{
    std::map<int, std::set<RegionId>> out;
    const auto layout   = w.terrain().layout(0);
    const double ground = w.terrainHeight(0, camera.x(), camera.z());
    for (const RegionId& r : desiredTestTiles(layout, *w.terrain().pyramid(0), camera, 0, 0, ground))
        out[r.level].insert(r);
    (void)s;
    return out;
}
}  // namespace

TEST(TerrainStreaming, ResidentTileEqualsHook)
{
    World w(terrainSetup(16, 1, ruleHook));
    RecordingRenderSink sink;
    std::mt19937_64 rng(7);
    std::vector<std::pair<double, double>> points;
    for (int k = 0; k < 2000; ++k)
        points.push_back({double(rng() % (4096 * 64)) / 64 + 200, double(rng() % (600 * 64)) / 64 + 100});
    std::vector<double> before;
    for (auto [x, z] : points) before.push_back(w.terrainHeight(0, x, z));
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, Vec3(400, 50, 400)));
    ASSERT_GT(w.terrain().store(0).tilesAtLevel(0), 0);
    for (size_t i = 0; i < points.size(); ++i)
        ASSERT_EQ(w.terrainHeight(0, points[i].first, points[i].second), before[i]) << i;
    // Bitwise at the samples too
    for (const auto& [r, t] : w.terrain().store(0).tiles())
    {
        if (r.level != 0) continue;
        for (int j = 0; j < 257; j += 13)
            for (int i = 0; i < 257; i += 11)
                ASSERT_EQ(w.terrain().sample(0, r.x * 256 + i, r.z * 256 + j),
                          clampTerrainHeight(ruleHook(w.spaceSeed(0), r.x * 256 + i, r.z * 256 + j)));
    }
    w.setRenderSink(nullptr);
}

TEST(TerrainStreaming, FlatSpaceRequestsNothing)
{
    WorldSetup s;
    s.profiling = false;
    World w(s);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    w.terrain().updateView(0, Vec3(10, 10, 10));
    EXPECT_TRUE(w.terrain().completeView(0, Vec3(10, 10, 10)));
    EXPECT_FALSE(w.terrain().poolCreated());
    EXPECT_EQ(w.terrain().readiness(0, Vec3(10, 10, 10), 256), 1.0);
    EXPECT_FALSE(sink.hasTerrainSet());
    EXPECT_EQ(sink.counters().setTerrainPyramid, 0u);
    w.setRenderSink(nullptr);
}

TEST(TerrainStreamingDeathTest, GenerationExceptionIsFatal)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH(
        {
            World w(terrainSetup(
                16, 2, [](uint64_t, int64_t, int64_t) -> int32_t { throw std::runtime_error("the hook failed"); }));
            RecordingRenderSink sink;
            w.setRenderSink(&sink);
            w.terrain().completeView(0, Vec3(100, 10, 100), 5);
        },
        "terrain pyramid space 0: the hook failed");
}

TEST(TerrainStreaming, DesiredSetMatchesTheRule)
{
    World w(terrainSetup(32, 1, slopeRidgeHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    for (const Vec3& cam : {Vec3(1000, 100, 2000), Vec3(4000, 600, 7000), Vec3(10, 30, 8000)})
    {
        ASSERT_TRUE(w.terrain().completeView(0, cam));
        const auto ref = referenceDesired(w, cam, kDefaults);
        for (int k = 0; k < 5; ++k)
        {
            std::set<RegionId> got;
            for (const auto& [r, t] : sink.terrainTiles())
                if (r.level == k && t.heights) got.insert(r);
            // The sink holds the desired tiles (and older resident ones within the budget)
            for (const RegionId& r : ref.count(k) ? ref.at(k) : std::set<RegionId>())
                EXPECT_TRUE(got.count(r)) << "level " << k << " (" << r.x << ", " << r.z << ")";
            EXPECT_EQ(w.terrain().stats().levels[size_t(k)].desired, int(ref.count(k) ? ref.at(k).size() : 0));
        }
    }
    w.setRenderSink(nullptr);
}

// Regression: residency measured the 3D distance to the tile box while TerrainPass selects by the horizontal distance
// and the camera height above the ground; a cliff 500 m above the camera 304 m away was never requested at level 0
// (box distance 530 m > 400 m) although the pass refines into it (the residency border of a refused node)
TEST(TerrainStreaming, DesiredSetUsesTheLodDistanceAtACliff)
{
    World w(terrainSetup(32, 1, [](uint64_t, int64_t x, int64_t) { return x >= 2304 ? int32_t(500 * 1024) : 0; }));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    const Vec3 cam(2000, 2, 2000);
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    // The plateau's level-0 tile next to the camera's
    EXPECT_TRUE(sink.terrainTiles().count(tileRegion(0, 9, 7)));
    const auto ref = referenceDesired(w, cam, kDefaults);
    ASSERT_TRUE(ref.count(0));
    EXPECT_TRUE(ref.at(0).count(tileRegion(0, 9, 7)));
    for (int k = 0; k < 5; ++k)
        EXPECT_EQ(w.terrain().stats().levels[size_t(k)].desired, int(ref.count(k) ? ref.at(k).size() : 0)) << k;
    w.setRenderSink(nullptr);
}

// Regression (CodeRabbit, S20): the search started at floor((camera - range) / size), so the tile that ends exactly at
// camera.x - range (distance = range on flat ground at the camera height) was never visited, although it is desired
TEST(TerrainStreaming, DesiredSetIncludesTheTileAtTheRangeBorder)
{
    World w(terrainSetup(16, 2, flatHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    // 912 - 400 = 512: the tile (1, 1) ends at x = 512, z = 300 lies in its span; on the ground (height 0)
    const Vec3 cam(912, 0, 300);
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    EXPECT_TRUE(sink.terrainTiles().count(tileRegion(0, 1, 1)));
    const auto ref = referenceDesired(w, cam, kDefaults);
    ASSERT_TRUE(ref.count(0) && ref.at(0).count(tileRegion(0, 1, 1)));
    for (int k = 0; k < 5; ++k)
        EXPECT_EQ(w.terrain().stats().levels[size_t(k)].desired, int(ref.count(k) ? ref.at(k).size() : 0)) << k;
    w.setRenderSink(nullptr);
}

// Regression (CodeRabbit, S20): with residencyRange0 = 0 the request priority of the tile under the camera was
// int(0 / 0) (UB); only the tiles at distance 0 are desired and stream
TEST(TerrainStreaming, ZeroResidencyRange)
{
    World w(terrainSetup(16, 2, flatHook));
    TerrainViewSettings s = kDefaults;
    s.residencyRange0     = 0;
    w.terrain().setViewSettings(s);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, Vec3(300, 0, 300)));
    EXPECT_EQ(w.terrain().stats().levels[0].desired, 1);
    EXPECT_TRUE(sink.terrainTiles().count(tileRegion(0, 1, 1)));
    w.setRenderSink(nullptr);
}

// The grass masks keep the 3D distance to the chunk's box (the grass fades by its 3D distance): 60 m above flat ground
// the chunk (0, 0) 62 m away horizontally is within the 80 m radius (its box reaches 64 m up), although the residency's
// distance with the camera height (82 m) is not
TEST(TerrainStreaming, GrassMasksUseTheBoxDistance)
{
    World w(terrainSetup(16, 2, flatHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, Vec3(300, 60, 300)));
    ChunkId c;
    c.space = 0;
    c.x     = 0;
    c.z     = 0;
    EXPECT_TRUE(sink.grassMasks().count(c));
    w.setRenderSink(nullptr);
}

TEST(TerrainStreaming, DesiredSetTruncatedToBudget)
{
    World w(terrainSetup(64, 1, integerTestHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    TerrainViewSettings s = kDefaults;
    s.residencyRange0     = 2000;
    s.grassRadius         = 600;
    w.terrain().setViewSettings(s);
    const Vec3 cam(8000, 50, 8000);
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    const auto& st = w.terrain().stats();
    for (int k = 0; k < 5; ++k)
    {
        EXPECT_LE(st.levels[size_t(k)].desired, s.tilesPerLevel);
        EXPECT_LE(st.levels[size_t(k)].resident, s.tilesPerLevel);
        EXPECT_LE(sink.terrainTilesAtLevel(k), s.tilesPerLevel);
    }
    EXPECT_GT(st.levels[0].truncated, 0);
    // The nearest ones: the camera's chunk is resident
    EXPECT_TRUE(sink.terrainTiles().count(tileRegion(0, 31, 31)));
    EXPECT_LE(st.masksDesired, s.grassMaskLayers);
    EXPECT_GT(st.masksTruncated, 0);
    EXPECT_LE(int(sink.grassMasks().size()), s.grassMaskLayers);
    w.setRenderSink(nullptr);
}

// LRU eviction never removes a desired tile and keeps tilesPerLevel; removals are sent in the same update (the
// recording's tiles stay a subset of the store's); at most uploadsPerFrame additions per update
TEST(TerrainStreaming, EvictionAndUploadsPerFrame)
{
    World w(terrainSetup(64, 2, integerTestHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    TerrainViewSettings s = kDefaults;
    s.tilesPerLevel       = 12;
    s.uploadsPerFrame     = 3;
    w.terrain().setViewSettings(s);
    Vec3 cam(2000, 80, 2000);
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    for (int step = 0; step < 300; ++step)
    {
        cam.x() += 40;
        const uint64_t before = sink.counters().setTerrainTile;
        w.terrain().updateView(0, cam);
        EXPECT_LE(sink.counters().setTerrainTile - before, 3u);
        const auto& store = w.terrain().store(0);
        for (int k = 0; k < 4; ++k) ASSERT_LE(store.tilesAtLevel(k), s.tilesPerLevel);
        for (const auto& [r, t] : sink.terrainTiles()) ASSERT_TRUE(store.tile(r)) << "the sink holds a dropped tile";
        // Every desired tile in the store stays (the desired set of this camera)
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    EXPECT_GT(w.terrain().stats().evicted, 0u);
    w.setRenderSink(nullptr);
}

TEST(TerrainStreaming, UpdateViewRequestsThePyramidFirst)
{
    // The world first: the latch opens before the world joins its workers
    World w(terrainSetup(16, 2, latchedHook));
    std::optional<ScopedLatch> latch;
    latch.emplace();
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    w.terrain().updateView(0, Vec3(500, 30, 500));
    EXPECT_TRUE(w.terrain().poolCreated());
    EXPECT_EQ(w.terrain().stats().pending, 1);
    for (int k = 0; k < 8; ++k) EXPECT_EQ(w.terrain().stats().levels[size_t(k)].desired, 0);
    EXPECT_EQ(w.terrain().readiness(0, Vec3(500, 30, 500), 256), 0.0);
    w.terrain().updateView(0, Vec3(500, 30, 500));
    EXPECT_EQ(w.terrain().stats().pending, 1);
    EXPECT_EQ(sink.counters().setTerrainPyramid, 0u);
    latch.reset();
    // The pyramid arrives, then the tiles
    for (int i = 0; i < 2000 && sink.counters().setTerrainTile == 0; ++i)
    {
        w.terrain().updateView(0, Vec3(500, 30, 500));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(sink.counters().setTerrainPyramid, 1u);
    EXPECT_GT(sink.counters().setTerrainTile, 0u);
    w.setRenderSink(nullptr);
}

// One completeView on a fresh world: the pyramid, every desired tile and mask sent; readiness 1; a second call
// generates nothing
TEST(TerrainStreaming, CompleteViewFirstView)
{
    World w(terrainSetup(32, 1, slopeRidgeHook, 0));
    RecordingRenderSink sink;
    sink.setLogging(true);
    w.setRenderSink(&sink);
    const Vec3 cam(3000, 100, 3000);
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    const auto& st = w.terrain().stats();
    EXPECT_EQ(st.pyramidsGenerated, 1u);
    EXPECT_EQ(st.pending, 0);
    for (int k = 0; k < 5; ++k)
    {
        EXPECT_EQ(st.levels[size_t(k)].sent, st.levels[size_t(k)].desired) << k;
        EXPECT_GT(st.levels[size_t(k)].desired, 0) << k;
    }
    EXPECT_GT(st.masksDesired, 0);
    EXPECT_EQ(st.masksSent, st.masksDesired);
    EXPECT_EQ(w.terrain().readiness(0, cam, 1e9), 1.0);
    // The pyramid came first
    ASSERT_FALSE(sink.log().empty());
    EXPECT_EQ(sink.log().front(), "setTerrainPyramid 0");
    const uint64_t generated = st.generated;
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    EXPECT_EQ(w.terrain().stats().generated, generated);
    w.setRenderSink(nullptr);
}

// With the workers held and a 0.1 s timeout: false soon after 0.1 s (one caller job may finish), far before the
// default timeout; no fatal error
TEST(TerrainStreaming, CompleteViewTimeout)
{
    World w(terrainSetup(32, 1, latchedHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    // The first view with the latch open (the pyramid)
    ASSERT_TRUE(w.terrain().completeView(0, Vec3(1500, 100, 1500), 60));
    // A new area with held workers: the caller runs the queued jobs until the deadline
    std::optional<ScopedLatch> latch;
    latch.emplace();
    const Vec3 far(7000, 100, 7000);
    const auto t0  = std::chrono::steady_clock::now();
    const bool ok  = w.terrain().completeView(0, far, 0.1);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_FALSE(ok);
    // 0.1 s plus one caller job (a 1 m tile of the latched hook takes a few ms; generous for a loaded machine): no
    // caller job starts after the deadline
    EXPECT_LT(s, 2.0);
    // (That the caller runs queued jobs is CompleteViewRunsJobsOnTheCaller: a paused test thread may reach the short
    // deadline before its first job)
    // The workers take the queued jobs and block on the latch (on a loaded machine maybe only after the call returned)
    const auto blockedBy = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (latchState().callsBlocked.load() == 0 && std::chrono::steady_clock::now() < blockedBy)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_GT(latchState().callsBlocked.load(), 0);
    latch.reset();
    // Completes afterwards
    EXPECT_TRUE(w.terrain().completeView(0, far, 60));
    w.setRenderSink(nullptr);
}

// With the workers held, the caller runs the queued tile jobs itself: the view completes once the latch opens, which
// happens only after the caller ran a tile job (a helper thread waits for that)
TEST(TerrainStreaming, CompleteViewRunsJobsOnTheCaller)
{
    World w(terrainSetup(32, 1, latchedHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, Vec3(1500, 100, 1500), 60));
    std::optional<ScopedLatch> latch;
    latch.emplace();
    const Vec3 far(7000, 100, 7000);
    // The hook calls of one height query at far (no tile there yet): computeDesired makes the same once
    w.terrain().height(0, far.x(), far.z());
    const int heightCalls = latchState().callsOnArmingThread.exchange(0);
    // Regression (CodeRabbit, S20): the camera height query alone passed a "> 0" check; the caller ran tile jobs.
    // Moved out of CompleteViewTimeout (CodeRabbit, S20): its 0.1 s deadline may pass before the caller's first job
    std::atomic<bool> callerRan{false};
    std::thread opener(
        [&]()
        {
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (latchState().callsOnArmingThread.load() <= heightCalls && std::chrono::steady_clock::now() < until)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            callerRan = latchState().callsOnArmingThread.load() > heightCalls;
            openLatch();
        });
    const bool ok = w.terrain().completeView(0, far, 60);
    opener.join();
    EXPECT_TRUE(ok);
    EXPECT_TRUE(callerRan);
    latch.reset();
    w.setRenderSink(nullptr);
}

// Regression: completeView spun until its deadline (120 s by default) when nothing could progress; with the requests
// held it returns false at once
TEST(TerrainStreaming, CompleteViewWithoutProgressReturnsAtOnce)
{
    World w(terrainSetup(32, 1, integerTestHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, Vec3(1500, 100, 1500)));
    w.terrain().holdRequestsForTests(true);
    const auto t0  = std::chrono::steady_clock::now();
    const bool ok  = w.terrain().completeView(0, Vec3(7000, 100, 7000));
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_FALSE(ok);
    EXPECT_LT(s, 10.0);
    EXPECT_EQ(w.terrain().stats().pending, 0);
    EXPECT_FALSE(w.terrain().viewComplete(0, Vec3(7000, 100, 7000)));
    // Released, it completes
    w.terrain().holdRequestsForTests(false);
    EXPECT_TRUE(w.terrain().completeView(0, Vec3(7000, 100, 7000)));
    w.setRenderSink(nullptr);
}

// Regression: a new sink received the whole store after setViewSettings lowered the budget (more tiles than the new
// layout's layers: "no free layer"); it receives the last view's desired set of the current budget
TEST(TerrainStreaming, NewSinkAfterALowerBudget)
{
    World w(terrainSetup(32, 1, integerTestHook));
    TerrainViewSettings s = w.terrain().viewSettings();
    s.grassRadius         = 600;
    w.terrain().setViewSettings(s);
    RecordingRenderSink first;
    w.setRenderSink(&first);
    const Vec3 cam(4000, 60, 4000);
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    ASSERT_GT(first.terrainTilesAtLevel(0), 6);
    ASSERT_GT(int(first.grassMasks().size()), 4);
    s.tilesPerLevel   = 6;
    s.grassMaskLayers = 4;
    w.terrain().setViewSettings(s);
    RecordingRenderSink second;
    w.setRenderSink(&second);
    EXPECT_EQ(second.counters().setTerrainPyramid, 1u);
    for (int k = 0; k < 5; ++k) EXPECT_LE(second.terrainTilesAtLevel(k), 6) << k;
    EXPECT_GT(second.terrainTilesAtLevel(0), 0);
    EXPECT_LE(int(second.grassMasks().size()), 4);
    w.setRenderSink(nullptr);
}

// Regression: a non-finite camera reached integer conversions; completeView and updateView refuse it at once and
// request nothing
TEST(TerrainStreaming, NonFiniteCameraRequestsNothing)
{
    World w(terrainSetup(32, 1, integerTestHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto t0    = std::chrono::steady_clock::now();
    EXPECT_FALSE(w.terrain().completeView(0, Vec3(nan, 100, 100)));
    EXPECT_FALSE(w.terrain().completeView(0, Vec3(100, std::numeric_limits<double>::infinity(), 100)));
    w.terrain().updateView(0, Vec3(100, 100, nan));
    EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 5.0);
    EXPECT_FALSE(w.terrain().poolCreated());
    EXPECT_EQ(w.terrain().stats().pending, 0);
    EXPECT_EQ(sink.counters().setTerrainPyramid, 0u);
    EXPECT_EQ(sink.counters().setTerrainTile, 0u);
    w.setRenderSink(nullptr);
}

// Masks for chunks within the grass radius, removed when they leave it, regenerated after a threshold change
TEST(TerrainStreaming, GrassMasks)
{
    World w(terrainSetup(32, 2, slopeRidgeHook, 0));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    Vec3 cam(1000, 40, 1000);
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    const auto masks = sink.grassMasks();
    ASSERT_FALSE(masks.empty());
    for (const auto& [c, m] : masks)
    {
        const double dx = std::max({c.x * 256.0 - cam.x(), 0.0, cam.x() - (c.x + 1) * 256.0});
        const double dz = std::max({c.z * 256.0 - cam.z(), 0.0, cam.z() - (c.z + 1) * 256.0});
        EXPECT_LE(std::hypot(dx, dz), 80.0 + 1e-9);
    }
    // Away: the masks leave
    cam = Vec3(5000, 40, 5000);
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    for (const auto& [c, m] : masks) EXPECT_FALSE(sink.grassMasks().count(c));
    EXPECT_GT(sink.counters().removeGrassMask, 0u);
    // A threshold change: every mask is dropped and regenerated
    const uint64_t sent   = sink.counters().setGrassMask;
    TerrainViewSettings s = kDefaults;
    s.grassMinNormalY     = 0.95;
    w.terrain().setViewSettings(s);
    EXPECT_TRUE(sink.grassMasks().empty());
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    EXPECT_GT(sink.counters().setGrassMask, sent);
    EXPECT_FALSE(sink.grassMasks().empty());
    w.setRenderSink(nullptr);
}

// Changing tilesPerLevel sends a pyramid with the new layout; the recording holds only tiles sent after it, within the
// new budget
TEST(TerrainStreaming, LayoutChangeStartsNewSet)
{
    World w(terrainSetup(32, 1, integerTestHook));
    RecordingRenderSink sink;
    sink.setLogging(true);
    w.setRenderSink(&sink);
    const Vec3 cam(2000, 100, 2000);
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    EXPECT_GT(sink.terrainTilesAtLevel(0), 10);
    TerrainViewSettings s = kDefaults;
    s.tilesPerLevel       = 6;
    w.terrain().setViewSettings(s);
    // Regression (CodeRabbit, S20): readiness counted the smaller set as sent although the sink held the old layout
    EXPECT_FALSE(w.terrain().viewComplete(0, cam));
    EXPECT_EQ(w.terrain().readiness(0, cam, 1e9), 0.0);
    sink.clearLog();
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    ASSERT_FALSE(sink.log().empty());
    EXPECT_EQ(sink.log().front(), "setTerrainPyramid 0");
    EXPECT_EQ(sink.terrainLayout().tilesPerLevel, 6);
    for (int k = 0; k < 5; ++k) EXPECT_LE(sink.terrainTilesAtLevel(k), 6);
    // Masks re-sent too
    EXPECT_FALSE(sink.grassMasks().empty());
    EXPECT_EQ(w.terrain().readiness(0, cam, 1e9), 1.0);
    w.setRenderSink(nullptr);
}

// Regression (CodeRabbit, S20): a sink attached again still held the tiles of its old view (no removal calls reach a
// detached sink); the pyramid of the same space and layout kept them and the replay of the new view overflowed
// tilesPerLevel. Now the reattached sink gets clearTerrain first
TEST(TerrainStreaming, ReattachedSinkDropsItsOldSet)
{
    World w(terrainSetup(32, 1, integerTestHook));
    TerrainViewSettings s = kDefaults;
    s.tilesPerLevel       = 1;
    w.terrain().setViewSettings(s);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, Vec3(1000, 100, 1000)));
    const auto oldTiles = sink.terrainTiles();
    w.setRenderSink(nullptr);
    const Vec3 far(7000, 100, 7000);
    ASSERT_TRUE(w.terrain().completeView(0, far));
    w.setRenderSink(&sink);
    EXPECT_EQ(sink.counters().clearTerrain, 1u);
    for (int k = 0; k < 5; ++k) EXPECT_LE(sink.terrainTilesAtLevel(k), 1) << k;
    for (const auto& t : sink.terrainTiles())
    {
        if (t.first.level == 0)
        {
            EXPECT_EQ(oldTiles.count(t.first), 0u);
        }
    }
    EXPECT_TRUE(w.terrain().viewComplete(0, far));
    w.setRenderSink(nullptr);
}

// Regression (CodeRabbit, S20): the upload count of the last update stayed after an update that returned early (here
// a non-finite camera) and sent nothing
TEST(TerrainStreaming, UploadCountResetByEveryUpdate)
{
    World w(terrainSetup(16, 2, integerTestHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, Vec3(500, 40, 500)));
    // Uploads of an update (the tiles of a far camera, at most uploadsPerFrame per update)
    for (int i = 0; i < 2000 && w.terrain().stats().uploadsLastUpdate == 0; ++i)
    {
        w.terrain().updateView(0, Vec3(3500, 40, 3500));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GT(w.terrain().stats().uploadsLastUpdate, 0);
    w.terrain().updateView(0, Vec3(std::numeric_limits<double>::quiet_NaN(), 40, 500));
    EXPECT_EQ(w.terrain().stats().uploadsLastUpdate, 0);
    w.setRenderSink(nullptr);
}

TEST(TerrainStreaming, Readiness)
{
    World w(terrainSetup(32, 1, integerTestHook));
    RecordingRenderSink sink;
    EXPECT_EQ(w.terrain().terrainUploadsPending(), 0u);
    w.setRenderSink(&sink);
    sink.uploadsPendingForTests = 4096;
    EXPECT_EQ(w.terrain().terrainUploadsPending(), 4096u);
    sink.uploadsPendingForTests = 0;
    const Vec3 cam(2500, 80, 2500);
    EXPECT_EQ(w.terrain().readiness(0, cam, 256), 0.0);
    double last = 0;
    for (int i = 0; i < 4000; ++i)
    {
        w.terrain().updateView(0, cam);
        const double r = w.terrain().readiness(0, cam, 256);
        EXPECT_GE(r, last);
        last = r;
        if (r == 1.0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(last, 1.0);
    ASSERT_TRUE(w.terrain().completeView(0, cam));
    EXPECT_EQ(w.terrain().readiness(0, cam, 1e9), 1.0);
    w.setRenderSink(nullptr);
    // No terrain: 1
    WorldSetup flat;
    flat.profiling = false;
    World f(flat);
    EXPECT_EQ(f.terrain().readiness(0, cam, 256), 1.0);
}

// clearTerrain drops the recording's set; a later pyramid starts a new one
TEST(TerrainStreaming, ClearTerrainOnRecording)
{
    World w(terrainSetup(16, 2, integerTestHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, Vec3(500, 40, 500)));
    ASSERT_TRUE(sink.hasTerrainSet());
    EXPECT_FALSE(sink.terrainTiles().empty());
    EXPECT_FALSE(sink.grassMasks().empty());
    sink.clearTerrain();
    EXPECT_FALSE(sink.hasTerrainSet());
    EXPECT_TRUE(sink.terrainTiles().empty());
    EXPECT_TRUE(sink.grassMasks().empty());
    EXPECT_EQ(sink.terrainPyramid(), nullptr);
    sink.setTerrainPyramid(0, *w.terrain().pyramid(0));
    EXPECT_TRUE(sink.hasTerrainSet());
    EXPECT_EQ(sink.counters().clearTerrain, 1u);
    w.setRenderSink(nullptr);
}

// A world with a recording sink (as every scenario world) and no updateView / completeView streams nothing; without a
// sink updateView sends nothing; a sink set after a completeView receives the pyramid, tiles and masks
TEST(TerrainStreaming, SinkAloneStreamsNothing)
{
    WorldSetup setup = terrainSetup(16, 2, slopeRidgeHook, 0);
    World w(setup);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    Entity e = character(w, Vec3(100, 0, 100), Vec3(2, 0, 1));
    w.step(100);
    EXPECT_FALSE(w.terrain().poolCreated());
    EXPECT_EQ(w.terrain().stats().pending, 0);
    EXPECT_FALSE(sink.hasTerrainSet());
    EXPECT_EQ(sink.counters().setTerrainTile, 0u);
    EXPECT_GT(w.read<Transform>(e).position().x(), 100);
    w.setRenderSink(nullptr);
    // No sink: updateView sends (and requests) nothing
    w.terrain().updateView(0, Vec3(100, 10, 100));
    EXPECT_FALSE(w.terrain().poolCreated());
    // completeView without a sink fills the store; a sink set afterwards receives it through the send point
    ASSERT_TRUE(w.terrain().completeView(0, Vec3(100, 10, 100)));
    RecordingRenderSink later;
    w.setRenderSink(&later);
    EXPECT_TRUE(later.hasTerrainSet());
    EXPECT_EQ(int(later.terrainTiles().size()), int(w.terrain().store(0).tiles().size()));
    EXPECT_EQ(later.grassMasks().size(), w.terrain().store(0).masks().size());
    EXPECT_FALSE(later.grassMasks().empty());
    w.setRenderSink(nullptr);
}

// Streaming changes neither the world's hash nor terrainHeight
TEST(TerrainStreaming, StreamingKeepsHashAndHeights)
{
    auto run = [](bool stream)
    {
        World w(terrainSetup(32, 1, ruleHook, 0));
        RecordingRenderSink sink;
        w.setRenderSink(&sink);
        Entity e = character(w, Vec3(1000, 0, 1000), Vec3(3, 0, 2));
        std::vector<double> ys;
        for (int i = 0; i < 120; ++i)
        {
            if (stream)
            {
                EXPECT_TRUE(w.terrain().completeView(0, w.read<Transform>(e).position() + Vec3(0, 20, 0))) << i;
            }
            w.step();
            ys.push_back(w.read<Transform>(e).position().y());
        }
        const uint64_t h = w.hash();
        w.setRenderSink(nullptr);
        return std::make_pair(h, ys);
    };
    const auto a = run(false), b = run(true);
    EXPECT_EQ(a.first, b.first);
    EXPECT_EQ(a.second, b.second);
}

// patchTerrainTile / patchTerrainPyramid through the sink change the recorded copies; an unsent region or another space
// is ignored and counted
TEST(TerrainStreaming, PatchesReachTheRecording)
{
    World w(terrainSetup(16, 2, integerTestHook));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, Vec3(300, 40, 300)));
    const RegionId r = tileRegion(0, 1, 1);
    ASSERT_TRUE(sink.terrainTiles().count(r));
    const auto storeHeights         = w.terrain().store(0).tile(r)->data.heights;
    const std::vector<int32_t> data = {1, 2, 3, 4, 5, 6};
    RenderSink& s                   = sink;
    s.patchTerrainTile(r, 10, 20, 3, 2, data.data());
    const auto& t = *sink.terrainTiles().at(r).heights;
    EXPECT_EQ(t[size_t(20 * 129 + 10)], 1);
    EXPECT_EQ(t[size_t(21 * 129 + 12)], 6);
    // Regression (CodeRabbit, S20): a patch whose input is the recorded (sole-owner) heights read them after the
    // assignment released them
    const int32_t* own = sink.terrainTiles().at(r).heights->data() + (20 * 129 + 10);
    s.patchTerrainTile(r, 0, 0, 1, 1, own);
    EXPECT_EQ((*sink.terrainTiles().at(r).heights)[0], 1);
    EXPECT_LE(sink.terrainTiles().at(r).minHeight, 1);
    // The store keeps the base data
    EXPECT_EQ((*storeHeights)[size_t(20 * 129 + 10)],
              integerTestHook(w.spaceSeed(0), 2 * (256 / 2 + 10), 2 * (128 + 20)));
    s.patchTerrainPyramid(0, 5, 6, 2, 3, data.data());
    EXPECT_EQ((*sink.terrainPyramid()->heights)[size_t(8 * 129 + 6)], 6);
    EXPECT_NE(w.terrain().pyramid(0)->heights, sink.terrainPyramid()->heights);
    s.patchTerrainTile(tileRegion(0, 15, 15), 0, 0, 1, 1, data.data());
    s.patchTerrainPyramid(3, 0, 0, 1, 1, data.data());
    EXPECT_EQ(sink.terrainPatchesIgnored(), 2u);
    w.setRenderSink(nullptr);
}

// §22 acceptance (S20): the terrain memory (CPU + GPU) of the game_world layout (128 chunks, 1 m, default budgets)
// stays <= 768 MB along the flyover circle, a 30 km line at 300 m and at 2 m, and a 90 degree turn
TEST(TerrainStreaming, MemoryBudget32km)
{
    World w(terrainSetup(128, 1, integerTestHook, 6144));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    const size_t budget           = size_t(768) << 20;
    const Vulkan::TerrainLayout L = w.terrain().layout(0);
    const auto est                = Vulkan::TerrainAtlas::memoryEstimate(L);
    // The static bound: the budgets times the layout, the pyramid, the masks, the results in flight
    const size_t n = size_t(L.samplesPerSide()), P = size_t(L.pyramidSamplesPerSide());
    const size_t staticBound = size_t(L.streamedLevels()) * size_t(L.tilesPerLevel) * n * n * 4 + P * P * 4 +
                               size_t(L.grassMaskLayers) * 256 * 256 + size_t(kDefaults.maxInFlight) * n * n * 4 +
                               est.gpuBytes() + est.cpuBytes();
    EXPECT_LE(staticBound, budget);
    size_t worst = 0;
    auto at      = [&](const Vec3& cam)
    {
        ASSERT_TRUE(w.terrain().completeView(0, cam));
        const size_t total = w.terrain().stats().cpuBytes + sink.terrainGpuBytes() + sink.terrainSceneCpuBytes();
        worst              = std::max(worst, total);
        ASSERT_LE(total, budget) << cam.transpose();
    };
    // The flyover circle (every 20th of 600 poses: centre 4096, radius 2458 m, 300 m up)
    for (int i = 0; i < 600; i += 20)
    {
        const double a = 2 * kPi * i / 600.0;
        at(Vec3(4096 + 2458 * std::cos(a), 300, 4096 + 2458 * std::sin(a)));
    }
    // A 30 km straight line 300 m and 2 m above the ground
    for (double h : {300.0, 2.0})
        for (double x = 1000; x <= 31000; x += 1500) at(Vec3(x, w.terrainHeight(0, x, 16000) + h, 16000));
    // A 90 degree turn
    for (int i = 0; i <= 10; ++i)
    {
        const double a = kPi / 2 * i / 10.0;
        at(Vec3(16000 + 500 * std::cos(a), 150, 16000 + 500 * std::sin(a)));
    }
    std::cout << "MemoryBudget32km: worst " << worst / double(1 << 20) << " MB, static bound "
              << staticBound / double(1 << 20) << " MB" << std::endl;
    w.setRenderSink(nullptr);
}
