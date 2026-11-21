/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Non-gating content generation benchmarks (docs/engine_design.md §17.7, §17.10, R-TEST-2; S23a): the plan time of the
// test setup and of the prototype setup, the chunk generation time per section (single-threaded; the report target is
// at most 50 ms per 1 m chunk on the desktop reference), the phase-7 cost of chunkStampSystem with nothing to stamp
// and with one site entering, and terrain heights through the pipeline against S20's hook. Measured in the release
// build of AGENTS.md; printed only (the CI run checks that it runs).

#include "saiga/engine/components/Player.h"
#include "saiga/engine/content/ContentCache.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/content/gen/ChunkGeneration.h"
#include "saiga/engine/core/Profiler.h"
#include "saiga/engine/terrain/HookGenerator.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/World.h"
#include "saiga/vulkan/world/PrototypeTerrain.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "content_test_setup.h"

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
double nowMs()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class F>
double medianMs(int runs, F&& f)
{
    std::vector<double> t;
    for (int i = 0; i < runs; ++i)
    {
        const double a = nowMs();
        f();
        t.push_back(nowMs() - a);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

WorldSetup prototypeSetup()
{
    WorldSetup s;
    s.seed = 1;
    SpaceTemplate t;
    t.name                  = "prototype";
    t.baseSpacing           = 1;
    t.sizeChunks            = Vulkan::kPrototypeSpaceChunks;
    t.seaLevel              = Vulkan::kPrototypeSeaLevel;
    t.generator             = 1;
    s.config.spaceTemplates = {t};
    s.generators            = [](GeneratorList& l)
    {
        l.add(1,
              std::make_shared<HookGenerator>("prototype", Vulkan::prototypeHeight, HookGrassRule{0.82, 0.4}, false));
    };
    return s;
}
}  // namespace

TEST(ContentBenchmark, PlansAndChunks)
{
    const WorldSetup test  = ContentTest::contentTestSetup(2);
    const double planTest  = medianMs(5,
                                      [&]
                                      {
                                         ContentCache cache;
                                         ASSERT_TRUE(GenerationPipeline::build(test, {}, cache).ok());
                                     });
    const WorldSetup proto = prototypeSetup();
    const double planProto = medianMs(5,
                                      [&]
                                      {
                                          ContentCache cache;
                                          ASSERT_TRUE(GenerationPipeline::build(proto, {}, cache).ok());
                                      });
    std::cout << "plan time: test setup " << planTest << " ms (3 spaces, 68 sites), prototype setup " << planProto
              << " ms" << std::endl;

    ContentCache cache;
    const GenerationPlans plans = GenerationPipeline::build(test, {}, cache);
    ASSERT_TRUE(plans.ok());
    for (SpaceId space : {SpaceId(0), SpaceId(1)})
    {
        const SpaceGeneration& sp = *plans.spaces[space];
        ChunkInputs in;
        in.generator     = sp.generator.get();
        in.pinnedVersion = sp.pinned;
        in.plan          = sp.plan.get();
        in.space         = space;
        in.sites         = &sp.siteContent;
        // Site 0's chunk of each space
        ChunkId c;
        c.space = space;
        c.x     = space == 0 ? 4 : 3;
        c.z     = space == 0 ? 5 : 4;
        ChunkBase base;
        const double heights = medianMs(3, [&] { base = generateChunkBase(in, c); });
        GrassMask mask;
        const double grass     = medianMs(3, [&] { sp.generator->grassMask(sp.current, *sp.plan, c, mask); });
        const double placement = medianMs(3, [&] { generateChunkPlacement(in, base, sp.current, 0, 0); });
        std::cout << "chunk generation (" << sp.settings.baseSpacing << " m, single-threaded): heights " << heights
                  << " ms, tier1 " << std::max(0.0, placement - grass) << " ms, grass " << grass << " ms, total "
                  << heights + placement << " ms" << std::endl;
    }
}

TEST(ContentBenchmark, StampSystemAndHeights)
{
    WorldSetup s = ContentTest::contentTestSetup(2);
    s.profiling  = true;
    World w(s);
    const int scope = w.profiler().scope("step/phase:streaming/system:chunk_stamp");
    const Entity p  = w.createIn(0);
    w.get<Transform>(p).setPosition(Vec3(20000, 0, 20000));
    w.add<Player>(p);
    std::vector<double> idle;
    for (int i = 0; i < 50; ++i)
    {
        w.step();
        double ms = 0;
        if (w.profiler().lastSample(scope, ms)) idle.push_back(ms);
    }
    std::sort(idle.begin(), idle.end());
    w.get<Transform>(p).setPosition(Vec3(1100, 0, 1200));
    w.step();
    double entering = 0;
    w.profiler().lastSample(scope, entering);
    std::cout << "chunkStampSystem (fullRadius 1000 m): nothing to stamp " << (idle.empty() ? 0 : idle[idle.size() / 2])
              << " ms (median), one site entering " << entering << " ms" << std::endl;

    // terrainHeight through the pipeline (the prototype HookGenerator) against S20's hook source, no resident tiles
    World proto(prototypeSetup());
    const TerrainHeightSource hook = TerrainHeightSource::fromHook(Vulkan::prototypeHeight, 1);
    const TerrainHeightSource pipe = proto.generation().heightSource(0);
    const int n                    = 200000;
    volatile int64_t sink          = 0;
    const double tHook             = medianMs(3,
                                              [&]
                                              {
                                      for (int i = 0; i < n; ++i) sink = sink + hook(i * 37 % 30000, i * 11 % 30000);
                                  });
    const double tPipe             = medianMs(3,
                                              [&]
                                              {
                                      for (int i = 0; i < n; ++i) sink = sink + pipe(i * 37 % 30000, i * 11 % 30000);
                                  });
    const double tWorld            = medianMs(3,
                                              [&]
                                              {
                                       double acc = 0;
                                       for (int i = 0; i < n; ++i)
                                           acc +=
                                               proto.terrainHeight(0, (i * 37 % 30000) + 0.3, (i * 11 % 30000) + 0.6);
                                       sink = sink + int64_t(acc);
                                   });
    std::cout << "height samples: S20 hook source " << tHook * 1e6 / n << " ns, pipeline source " << tPipe * 1e6 / n
              << " ns; World::terrainHeight " << tWorld * 1e6 / n << " ns" << std::endl;
    for (int i = 0; i < 1000; ++i) ASSERT_EQ(hook(i * 97, i * 13), pipe(i * 97, i * 13));
}
