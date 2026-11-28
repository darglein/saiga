/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Non-gating save benchmark (docs/engine_design.md §17.10, R-TEST-2; S25), memory file system, prints only: the freeze
// of a commit (the world-state serialization and the copy-on-write references) with resident entities and 50 edited
// 1 m chunks, the commit's write on the IO lane, unload and reload of chunks of 100 entities, save() of a world with
// stored groups on disk, and the open of edited chunks. CI (default): 10^4 entities, 100 stored groups, 200 edited
// chunks; local: SAIGA_SAVE_BENCHMARK=full: 10^5 entities, 1000 stored groups, 1000 edited chunks (release build:
// AGENTS.md's benchmark recipe).

#include "saiga/core/util/env.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/save/MemorySaveFileSystem.h"
#include "saiga/engine/save/SaveDirectory.h"
#include "saiga/engine/terrain/TerrainBrush.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "unload_test_setup.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace UnloadTest;

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
    return GetEnv("SAIGA_SAVE_BENCHMARK") == "full";
}

// The unload setup with 1 m base spacing (the largest deltas per chunk)
WorldSetup benchmarkSetup()
{
    WorldSetup s = unloadTestSetup();
    for (SpaceTemplate& t : s.config.spaceTemplates) t.baseSpacing = 1;
    return s;
}

TerrainBrush brush(double x, double z, double radius)
{
    TerrainBrush b;
    b.op       = TerrainBrush::Add;
    b.center   = Vec3(x, 0, z);
    b.radius   = radius;
    b.strength = 3;
    return b;
}

SaveDirectoryOptions memOptions(std::shared_ptr<SaveFileSystem> fs, bool synchronous)
{
    SaveDirectoryOptions o;
    o.fileSystem            = std::move(fs);
    o.synchronousIo         = synchronous;
    o.commitIntervalSeconds = 0;
    return o;
}
}  // namespace

TEST(SaveBenchmark, FreezeAndWrite)
{
    const bool full = fullRun();
    const int count = full ? 100000 : 10000;
    World w(benchmarkSetup());
    w.setStreamingSynchronous(true);
    addPlayer(w, Vec3(4096, 0, 4096));
    // Entities around the player (resident, inside the kept distance)
    Rng rng(1);
    for (int i = 0; i < count; ++i)
    {
        const Entity e = w.create();
        w.get<Transform>(e).teleport(Vec3(4096 + rng.uniform() * 200 - 100, 0, 4096 + rng.uniform() * 200 - 100),
                                     Quat::Identity());
    }
    // 50 edited 1 m chunks (a large brush per chunk)
    for (int i = 0; i < 50; ++i) w.editTerrain(0, brush(128 + 256.0 * (i % 25), 128 + 256.0 * (i / 25), 100));
    w.step();
    auto fs = std::make_shared<MemorySaveFileSystem>();
    ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs, false)).ok);
    ASSERT_TRUE(w.waitForCommits());
    const ChunkStore::Stats& s = w.chunkStoreView().stats();
    std::cout << "SaveBenchmark.FreezeAndWrite (" << (full ? "full" : "CI") << "): " << count << " entities, 50 edited "
              << "1 m chunks" << std::endl;
    std::cout << "  first commit: freeze " << s.lastFreezeMs << " ms, write " << s.lastCommitWriteMs << " ms ("
              << s.bytesWritten << " bytes, " << s.recordsWritten << " records)" << std::endl;
    // The 50 chunks edited again: the freeze holds block references (no delta copies)
    for (int i = 0; i < 50; ++i) w.editTerrain(0, brush(128 + 256.0 * (i % 25), 128 + 256.0 * (i / 25), 90));
    const uint64_t bytes = s.bytesWritten;
    const auto b0        = Clock::now();
    const size_t blob    = w.save().size();
    std::cout << "  save() of the world (the blob, for comparison) " << ms(b0, Clock::now()) << " ms, " << blob
              << " bytes" << std::endl;
    ASSERT_EQ(w.commit(), 2u);
    const double freeze = s.lastFreezeMs;
    ASSERT_TRUE(w.waitForCommits());
    std::cout << "  second commit: freeze " << freeze << " ms (budget 10 ms with 10^5 entities), write "
              << s.lastCommitWriteMs << " ms (" << (s.bytesWritten - bytes) << " bytes on the pool)" << std::endl;
    EXPECT_EQ(w.saveStatus().committed, 2u);
}

TEST(SaveBenchmark, UnloadReloadAndSave)
{
    const bool full  = fullRun();
    const int groups = full ? 1000 : 100;
    World w(unloadTestSetup());
    w.setStreamingSynchronous(true);
    const Vec3 home(1000, 0, 1000), away(6000, 0, 6000);
    const Entity player = addPlayer(w, home);
    // Groups of 100 entities (a root with 99 children) on a 10 x 10 grid around the player
    for (int g = 0; g < groups; ++g)
    {
        const Vec3 at     = home + Vec3(-120 + 24.0 * (g % 10), 0, -120 + 24.0 * ((g / 10) % 10));
        const Entity root = w.spawn<Grower>(at);
        for (int c = 0; c < 99; ++c)
        {
            const Entity e = w.create();
            w.add<Parent>(e, Parent{root});
        }
    }
    w.step();
    const auto t0 = Clock::now();
    w.get<Transform>(player).teleport(away, Quat::Identity());
    w.step();
    const auto t1 = Clock::now();
    ASSERT_EQ(w.chunkStoreView().totalGroups(), size_t(groups));
    // save() with every group on disk (a directory; the chunk is not resident: the bytes left memory)
    auto fs = std::make_shared<MemorySaveFileSystem>();
    ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs, true)).ok);
    const auto t2                   = Clock::now();
    const std::vector<uint8_t> blob = w.save();
    const auto t3                   = Clock::now();
    w.get<Transform>(player).teleport(home, Quat::Identity());
    w.step();
    const auto t4 = Clock::now();
    ASSERT_EQ(w.chunkStoreView().totalGroups(), 0u);
    const double entities = double(groups) * 100;
    std::cout << "SaveBenchmark.UnloadReloadAndSave (" << (full ? "full" : "CI") << "): " << groups
              << " groups of 100 entities" << std::endl;
    std::cout << "  unload step " << ms(t0, t1) << " ms (" << ms(t0, t1) * 1000 / entities << " us per entity, "
              << ms(t0, t1) / double(groups) << " ms per 100-entity group)" << std::endl;
    std::cout << "  save() with the groups on disk " << ms(t2, t3) << " ms (" << blob.size() << " bytes, "
              << w.chunkStoreView().stats().syncReads << " synchronous reads)" << std::endl;
    std::cout << "  reload step (read, decode, create, hooks) " << ms(t3, t4) << " ms (" << ms(t3, t4) * 1000 / entities
              << " us per entity)" << std::endl;
}

TEST(SaveBenchmark, OpenEditedChunks)
{
    const bool full  = fullRun();
    const int chunks = full ? 1000 : 200;
    auto fs          = std::make_shared<MemorySaveFileSystem>();
    {
        World w(unloadTestSetup());
        w.setStreamingSynchronous(true);
        addPlayer(w, Vec3(100, 0, 100));
        for (int i = 0; i < chunks; ++i) w.editTerrain(0, brush(128 + 256.0 * (i % 32), 128 + 256.0 * (i / 32), 30));
        const auto t0 = Clock::now();
        ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs, true)).ok);
        std::cout << "SaveBenchmark.OpenEditedChunks (" << (full ? "full" : "CI") << "): " << chunks
                  << " edited chunks: the first commit " << ms(t0, Clock::now()) << " ms, "
                  << w.chunkStoreView().stats().bytesWritten << " bytes" << std::endl;
    }
    World v(unloadTestSetup());
    v.setStreamingSynchronous(true);
    const SaveOpenResult r = v.openSaveDirectory("mem", memOptions(fs, true));
    ASSERT_TRUE(r.ok) << r.error;
    std::cout << "  open " << r.openSeconds * 1000 << " ms, " << r.recordsRead << " records read" << std::endl;
    EXPECT_GE(r.recordsRead, size_t(chunks));
}
