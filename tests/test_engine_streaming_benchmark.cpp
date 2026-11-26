/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Non-gating streaming benchmark (docs/engine_design.md §17.7, R-TEST-2; S24, G21), headless: a streaming world on the
// prototype height hook at 1 m (a HookGenerator: no slots, the realistic base cost) with the asynchronous pool,
// entities spread uniformly over a 2.15 km square (10 % Drifters) and one player walking at 30 m/s through it. Reports
// the step time (median, p95, max) and its streaming / simulation / tick / behaviour parts, the simulated count, the
// resident bytes and the stalls; and the residency rebuild (cold and warm caches, the load). Asserts only correctness.
//
// CI (default): scaled radii (fullRadius 128, prefetchMargin 0, simulationRadius 64), 10^3 entities, 60 steps unpaced
// after a completeResidency. Local: SAIGA_STREAMING_BENCHMARK=full: the default radii (simulationRadius 384), 10^5
// entities, 3600 steps paced at 60 Hz (real time, 2 pool workers); the local acceptance is 0 streaming stalls and the
// step time within §17.10's 4 ms (release build).

#include "saiga/core/util/env.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/streaming/SimulationSet.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/World.h"
#include "saiga/vulkan/world/PrototypeTerrain.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "streaming_test_setup.h"
#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

bool fullRun()
{
    const std::string v = GetEnv("SAIGA_STREAMING_BENCHMARK");
    return v == "full" || v == "full-off";
}
// SAIGA_STREAMING_BENCHMARK=full-off: the same run without streaming (S23b's behaviour, G1: every entity simulated, no
// residency): the "before" numbers
bool streamingOff()
{
    return GetEnv("SAIGA_STREAMING_BENCHMARK") == "full-off";
}

struct Stat
{
    double median = 0, p95 = 0, max = 0;
};
Stat stat(std::vector<double> v)
{
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.median = v[v.size() / 2];
    s.p95    = v[std::min(v.size() - 1, size_t(std::ceil(0.95 * double(v.size()))) - 1)];
    s.max    = v.back();
    return s;
}
void print(const std::string& what, const Stat& s)
{
    std::cout << "  " << what << ": median " << s.median << " ms, p95 " << s.p95 << " ms, max " << s.max << " ms"
              << std::endl;
}

// 32 chunks (8 km) at 1 m on the prototype hook; the CI radii or the defaults
WorldSetup benchmarkSetup(bool full)
{
    WorldSetup s;
    s.seed = 9;
    SpaceTemplate t;
    t = TerrainTestHooks::testSpaceTemplate(32, 1, Vulkan::prototypeHeight);
    if (!full)
    {
        t.fullRadius       = 128;
        t.prefetchMargin   = 0;
        t.simulationRadius = 64;
    }
    s.config.spaceTemplates.push_back(t);
    s.config.streaming = !streamingOff();
    s.generators       = TerrainTestHooks::registerTestHookGenerators;
    s.classes          = StreamingTest::registerStreamingTestClasses;
    s.components       = StreamingTest::registerStreamingTestComponents;
    return s;
}

// The entities over the square [centre - 1075, centre + 1075]^2 (a seeded grid with jitter), every tenth a Drifter
void populate(World& w, int count, const Vec3& centre)
{
    Rng rng(17);
    const double half = 1075;
    for (int i = 0; i < count; ++i)
    {
        const Vec3 p(centre.x() - half + rng.uniform() * 2 * half, 0, centre.z() - half + rng.uniform() * 2 * half);
        if (i % 10 == 0)
            w.spawn<StreamingTest::Drifter>(p);
        else
            w.spawn<StreamingTest::Ticker>(p);
    }
}

double lastScope(World& w, const std::string& path)
{
    double v = 0;
    w.profiler().lastSample(w.profiler().scope(path), v);
    return v;
}
}  // namespace

TEST(StreamingBenchmark, StepTime)
{
    const bool full   = fullRun();
    const int count   = full ? 100000 : 1000;
    const int steps   = full ? 3600 : 60;
    const Vec3 centre = Vec3(4096, 0, 4096);
    World w(benchmarkSetup(full));
    populate(w, count, centre);
    // The player walks east through the square at 30 m/s
    const Vec3 start                               = centre - Vec3(full ? 900 : 15, 0, 0);
    const Entity player                            = StreamingTest::addPlayer(w, start);
    w.get<CharacterIntent>(player).desiredVelocity = Vec3(30, 0, 0);
    const auto t0                                  = Clock::now();
    ASSERT_TRUE(w.completeResidency(120));
    const double completeMs = ms(t0, Clock::now());
    w.step();
    std::vector<double> step, streaming, simulation, tick, behaviour, startScope, unloadPass;
    const double dt = 1.0 / 60;
    auto next       = Clock::now();
    for (int i = 0; i < steps; ++i)
    {
        if (full)
        {
            next += std::chrono::microseconds(int64_t(dt * 1e6));
            std::this_thread::sleep_until(next);
        }
        const auto a = Clock::now();
        w.step();
        step.push_back(ms(a, Clock::now()));
        streaming.push_back(lastScope(w, "step/phase:streaming/system:streaming"));
        simulation.push_back(lastScope(w, "step/phase:streaming/system:simulation"));
        tick.push_back(lastScope(w, "step/phase:transform/system:tick"));
        behaviour.push_back(lastScope(w, "step/phase:behaviour/system:behaviour"));
        startScope.push_back(lastScope(w, "step/streaming_start"));
        unloadPass.push_back(lastScope(w, "step/phase:streaming/system:chunk_entities"));
    }
    if (w.streaming()) w.simulationSet().countSimulated();
    const StreamingStats& s = w.residency().stats();
    std::cout << "StreamingBenchmark.StepTime (" << (full ? "full, paced at 60 Hz" : "CI, unpaced") << "): " << count
              << " entities, " << steps << " steps, player at 30 m/s" << std::endl;
    std::cout << "  completeResidency before the run: " << completeMs << " ms" << std::endl;
    print("step", stat(step));
    print("  streaming", stat(streaming));
    print("  simulation", stat(simulation));
    print("  step start (refresh, between-step evaluation)", stat(startScope));
    print("  tick (the cell reassignment spikes)", stat(tick));
    print("  behaviour", stat(behaviour));
    std::cout << "  simulated " << s.simulated << ", resident chunks " << s.residentChunks << ", resident bytes "
              << s.residentBytes << ", applied " << s.applied << ", sync built " << s.syncBuilt << ", evaluations "
              << s.evaluations << std::endl;
    std::cout << "  streaming stalls " << s.streamingStalls << " (player stall steps " << s.playerStallSteps
              << "), mover holds " << s.moverHolds << std::endl;
    // S25: the unloads and reloads (beyond the kept distance of the player) and the resident entities
    const ChunkStore::Stats& st = w.chunkStoreView().stats();
    const double seconds        = double(steps) * w.dt();
    std::cout << "  unloads " << st.groupsUnloaded << " (" << double(st.groupsUnloaded) / seconds << "/s), reloads "
              << st.groupsReloaded << " (" << double(st.groupsReloaded) / seconds << "/s), stored groups "
              << w.chunkStoreView().totalGroups() << ", resident entities " << w.registry().aliveCount() << std::endl;
    print("  chunk_entities (unload pass, reloads)", stat(unloadPass));
    // Correctness: the player moved and is simulated, the far entities are frozen
    EXPECT_GT(w.read<Transform>(player).position().x(), start.x());
    EXPECT_TRUE(w.simulated(player));
    if (streamingOff()) return;
    EXPECT_GT(s.simulated, 0);
    EXPECT_LT(s.simulated, int(w.registry().aliveCount()));
    if (full)
    {
        EXPECT_EQ(s.streamingStalls, 0u) << "the local acceptance (S24): 0 stalls";
    }
}

TEST(StreamingBenchmark, Rebuild)
{
    const bool full = fullRun();
    const Vec3 at(4096, 0, 4096);
    auto cache  = std::make_shared<ContentCache>();
    double cold = 0, warm = 0;
    std::vector<uint8_t> saved;
    {
        WorldSetup s    = benchmarkSetup(full);
        s.sharedContent = cache;
        World w(s);
        StreamingTest::addPlayer(w, at);
        const auto t0 = Clock::now();
        ASSERT_TRUE(w.completeResidency(120));
        cold  = ms(t0, Clock::now());
        saved = w.save();
    }
    WorldSetup s    = benchmarkSetup(full);
    s.sharedContent = cache;
    World w(s);
    StreamingTest::addPlayer(w, at);
    auto t0 = Clock::now();
    ASSERT_TRUE(w.completeResidency(120));
    warm = ms(t0, Clock::now());
    World loaded(s);
    t0 = Clock::now();
    ASSERT_TRUE(loaded.tryLoad(saved).ok());
    const double load = ms(t0, Clock::now());
    std::cout << "StreamingBenchmark.Rebuild (" << (full ? "full" : "CI") << " radii): completeResidency cold " << cold
              << " ms, warm cache " << warm << " ms; tryLoad " << load << " ms (the residency rebuild "
              << loaded.residency().stats().lastLoadRebuildMs << " ms, " << loaded.residency().stats().residentChunks
              << " chunks)" << std::endl;
    EXPECT_TRUE(loaded.chunkReady(0, at.x(), at.z()));
}
