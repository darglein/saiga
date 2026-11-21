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
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <random>
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
    EXPECT_DEATH(with({ok, ok}), "more than one space template");
    SpaceTemplate enclosed = ok;
    enclosed.kind          = SpaceKind::Enclosed;
    EXPECT_DEATH(with({enclosed}), "Enclosed");
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
}  // namespace

TEST(StreamingPool, PriorityOrderAndFifo)
{
    StreamingPool pool(1);
    Gate gate;
    std::vector<int> order;
    std::mutex orderMutex;
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
    armLatch();
    StreamingPool pool(1);
    std::atomic<int> value{0};
    pool.submit([&](const std::atomic<bool>&) { value = latchedHook(1, 2, 3); }, 0, nullptr);
    EXPECT_EQ(latchedHook(1, 2, 3), integerTestHook(1, 2, 3));
    EXPECT_FALSE(pool.waitIdleFor(0.05));
    EXPECT_EQ(value.load(), 0);
    openLatch();
    pool.waitIdle();
    EXPECT_EQ(value.load(), integerTestHook(1, 2, 3));
}
