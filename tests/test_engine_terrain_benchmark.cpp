/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Non-gating terraforming benchmark (docs/engine_design.md §17.7, R-TEST-2; S22), headless (no resident tiles: every
// gather and every read of an edited sample uses the height source): the game_world terraform schedule of 600 edits
// with a cheap integer hook and with the prototype hook (the worst case): the kernel alone, applyTerrainEdit with its
// hook reads, terrainSample of edited samples, the message encode + client decode / apply; hash() with 50 edited
// chunks, save / load and a late join of 50 full-chunk edited chunks at 1 m, one grass mask from merged heights, and
// the patch step on a recording sink with resident tiles. Prints milliseconds; asserts only correctness.

#include "saiga/engine/core/Log.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/terrain/BrushKernel.h"
#include "saiga/engine/terrain/TerrainEdit.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainGeneration.h"
#include "saiga/engine/terrain/TerrainMessages.h"
#include "saiga/engine/terrain/TerrainStore.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/terrain/TerrainVisualPatch.h"
#include "saiga/engine/world/World.h"
#include "saiga/vulkan/world/PrototypeTerrain.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace TerrainTestHooks;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Stat
{
    double median = 0, p95 = 0, total = 0;
};
Stat stat(std::vector<double> v)
{
    Stat s;
    if (v.empty()) return s;
    for (double x : v) s.total += x;
    std::sort(v.begin(), v.end());
    s.median = v[v.size() / 2];
    s.p95    = v[std::min(v.size() - 1, size_t(std::ceil(0.95 * double(v.size()))) - 1)];
    return s;
}
void print(const std::string& what, const Stat& s)
{
    std::cout << "  " << what << ": median " << s.median << " ms, p95 " << s.p95 << " ms, total " << s.total << " ms"
              << std::endl;
}

// The schedule of game_world --benchmark terraform (TerraformBenchmark): brush b = 2c + j on the corner c, orbiting
TerrainBrush scheduleBrush(int edit, const std::array<double, 4>& targets)
{
    struct Spec
    {
        TerrainBrush::Op op;
        double radius;
    };
    static const Spec specs[8] = {
        {TerrainBrush::Add, 4},  {TerrainBrush::Flatten, 12}, {TerrainBrush::Subtract, 8},  {TerrainBrush::Smooth, 16},
        {TerrainBrush::Add, 16}, {TerrainBrush::Flatten, 8},  {TerrainBrush::Subtract, 12}, {TerrainBrush::Smooth, 4},
    };
    const int b = edit % 8, c = b / 2, j = b % 2, n = edit / 8;
    const Spec s       = specs[b];
    const double orbit = 0.5 * s.radius;
    const double angle = j * 3.14159265358979323846 + double(n) / orbit;
    TerrainBrush brush;
    brush.op       = s.op;
    brush.radius   = s.radius;
    brush.strength = (s.op == TerrainBrush::Add || s.op == TerrainBrush::Subtract) ? 0.25 : 0.5;
    brush.center   = Vec3(3840 + 256 * (c % 2) + orbit * std::cos(angle), targets[size_t(c)],
                          3840 + 256 * (c / 2) + orbit * std::sin(angle));
    return brush;
}

// 32 chunks (the corners at 3840..4096 m lie inside), 1 m
WorldSetup setupWith(TerrainHeightHook hook, int spacing = 1, int32_t chunks = 32)
{
    WorldSetup s;
    s.profiling = false;
    s.seed      = 9;
    s.config.spaceTemplates.push_back(testSpaceTemplate(chunks, spacing, hook));
    s.generators = registerTestHookGenerators;
    return s;
}

// Fills the chunks (0..9, 0..4) with non-zero deltas (50 full-chunk edited chunks)
void fillFiftyChunks(World& w)
{
    TerrainEditStore& st = w.terrainEditsForUpdate();
    const int64_t S      = st.chunkSamples(0);
    // The outer border towards the chunks without deltas stays 0 (the border rule of a save)
    for (int64_t k = 0; k < 5 * S; ++k)
        for (int64_t i = 0; i < 10 * S; ++i)
        {
            const int32_t v = int32_t(1 + ((i * 7 + k * 13) % 2000));
            st.reserveFor(0, i, k, v);
            st.writeDelta(0, i, k, v);
        }
    for (int z = 0; z < 5; ++z)
        for (int x = 0; x < 10; ++x)
        {
            ChunkId c;
            c.x = x;
            c.z = z;
            st.bumpSequence(c);
        }
}
}  // namespace

TEST(TerrainEditBenchmark, Schedule)
{
    for (int hookIndex = 0; hookIndex < 2; ++hookIndex)
    {
        const TerrainHeightHook hook = hookIndex == 0 ? integerTestHook : Vulkan::prototypeHeight;
        std::cout << (hookIndex == 0 ? "integerTestHook" : "prototypeHeight") << " (600 edits, headless, 1 m)"
                  << std::endl;
        World w(setupWith(hook));
        std::array<double, 4> targets;
        for (int c = 0; c < 4; ++c) targets[size_t(c)] = w.terrainHeight(0, 3840 + 256 * (c % 2), 3840 + 256 * (c / 2));
        std::vector<double> kernel, apply, encode, decode, sampleMs;
        std::vector<uint64_t> hooks;
        uint64_t bytes = 0;
        // A client world for the decode / apply
        auto net      = LoopbackTransport::createNetwork(2);
        WorldSetup cs = setupWith(hook);
        cs.transport  = net[1].get();
        cs.role       = NetRole::Client;
        cs.localPeer  = 1;
        World client(cs);
        std::vector<int32_t> base, pre, out;
        for (int e = 0; e < 600; ++e)
        {
            const TerrainBrush b = scheduleBrush(e, targets);
            // The kernel alone (on the gathered heights)
            {
                const QuantizedBrush q    = BrushKernel::quantize(b);
                const SampleRect affected = BrushKernel::affectedRect(q, 1, w.terrainEdits().maxSample(0));
                const SampleRect read     = BrushKernel::readRect(q, affected, w.terrainEdits().maxSample(0));
                gatherSamples(w, 0, read, base, pre);
                out.resize(size_t(affected.width() * affected.height()));
                const auto t0 = Clock::now();
                BrushKernel::evaluate(q, 1, affected, read, pre.data(), out.data());
                kernel.push_back(ms(t0, Clock::now()));
            }
            const uint64_t h0 = w.terrain().baseHookSamples();
            const auto t0     = Clock::now();
            applyTerrainEdit(w, 0, b);
            apply.push_back(ms(t0, Clock::now()));
            hooks.push_back(w.terrain().baseHookSamples() - h0);
            // Encode as phase 8 would, decode and apply on the client
            const auto& list = w.replication().hostStateList();
            if (!list.empty())
            {
                std::vector<uint8_t> m;
                const auto e0 = Clock::now();
                encodeTerrainModification(0, list.back().space, list.back().parts, m);
                encode.push_back(ms(e0, Clock::now()));
                bytes += m.size();
                const auto d0 = Clock::now();
                client.replication().onTerrainModification(0, m.data(), m.size());
                decode.push_back(ms(d0, Clock::now()));
                w.replication().clearHostStates();
            }
        }
        // terrainSample of edited samples (the hook path: nothing resident)
        {
            const auto t0 = Clock::now();
            int64_t sum   = 0;
            int n         = 0;
            for (int c = 0; c < 4; ++c)
                for (int64_t k = -20; k <= 20; ++k)
                    for (int64_t i = -20; i <= 20; ++i, ++n)
                        sum += w.terrainSample(0, 3840 + 256 * (c % 2) + i, 3840 + 256 * (c / 2) + k);
            sampleMs.push_back(ms(t0, Clock::now()) / n * 1000);
            EXPECT_NE(sum, 0);
        }
        const TerrainEditStore::Stats& st = w.terrainEdits().stats();
        std::cout << "  " << st.edits << " edits, " << st.noopEdits << " no-ops, " << st.chunksEdited << " chunks, "
                  << st.changedSamples << " samples changed, store " << w.terrainEdits().memoryBytes() << " bytes"
                  << std::endl;
        print("kernel", stat(kernel));
        print("applyTerrainEdit", stat(apply));
        uint64_t hookTotal = 0;
        for (uint64_t h : hooks) hookTotal += h;
        std::cout << "  hook samples per edit: " << double(hookTotal) / 600.0 << std::endl;
        print("encode", stat(encode));
        print("client decode + apply", stat(decode));
        std::cout << "  terrainSample of an edited sample (hook path): " << sampleMs[0] << " us" << std::endl;
        std::cout << "  encoded bytes per second at 60 edits/s: " << double(bytes) / 10.0 << std::endl;
        EXPECT_GE(st.chunksEdited, 1200u);
        // The client ends with the host's state
        std::vector<int32_t> a, c;
        for (const auto& [id, e] : w.terrainEdits().chunks(0))
        {
            EXPECT_EQ(client.terrainEdits().sequence(id), e.sequence);
            w.terrainEdits().denseDelta(id, a);
            client.terrainEdits().denseDelta(id, c);
            EXPECT_EQ(a, c);
        }
    }
}

TEST(TerrainEditBenchmark, HashSaveLoadAndJoinOfFiftyChunks)
{
    World w(setupWith(Vulkan::prototypeHeight));
    fillFiftyChunks(w);
    EXPECT_EQ(w.terrainEdits().chunkCount(), 50u);
    auto t0             = Clock::now();
    const uint64_t h    = w.hash();
    const double hashMs = ms(t0, Clock::now());
    t0                  = Clock::now();
    const uint64_t h2   = w.hash();
    const double cached = ms(t0, Clock::now());
    EXPECT_EQ(h, h2);
    t0                              = Clock::now();
    const std::vector<uint8_t> blob = w.save();
    const double saveMs             = ms(t0, Clock::now());
    World b(setupWith(Vulkan::prototypeHeight));
    const uint64_t hooks    = b.terrain().baseHookSamples();
    t0                      = Clock::now();
    const LoadResult loaded = b.tryLoad(blob);
    ASSERT_TRUE(loaded.ok()) << loaded.error;
    const double loadMs = ms(t0, Clock::now());
    EXPECT_EQ(b.terrain().baseHookSamples(), hooks);  // the load calls no hook
    EXPECT_EQ(b.hash(), h);
    // A late join: the chunk snapshot encode on the host, the apply on the client
    auto net      = LoopbackTransport::createNetwork(2);
    WorldSetup hs = setupWith(Vulkan::prototypeHeight);
    hs.transport  = net[0].get();
    World host(hs);
    fillFiftyChunks(host);
    t0 = Clock::now();
    host.replication().sendChunkSnapshot(1);
    host.flushOutbox();
    const double encodeMs = ms(t0, Clock::now());
    WorldSetup cs         = setupWith(Vulkan::prototypeHeight);
    cs.transport          = net[1].get();
    cs.role               = NetRole::Client;
    cs.localPeer          = 1;
    World client(cs);
    const uint64_t clientHooks = client.terrain().baseHookSamples();
    t0                         = Clock::now();
    client.step();
    const double applyMs = ms(t0, Clock::now());
    EXPECT_EQ(client.terrainEdits().chunkCount(), 50u);
    EXPECT_EQ(client.terrain().baseHookSamples(), clientHooks);
    std::cout << "50 full-chunk edited chunks at 1 m: hash " << hashMs << " ms (cached " << cached << " ms), save "
              << saveMs << " ms (" << blob.size() << " bytes), load " << loadMs << " ms; join: snapshot encode "
              << encodeMs << " ms, client apply " << applyMs << " ms" << std::endl;
}

TEST(TerrainEditBenchmark, GrassMaskAndPatchStep)
{
    // One grass mask from merged heights (the pool job's cost)
    for (int spacing : {1, 2})
    {
        World w(setupWith(Vulkan::prototypeHeight, spacing));
        const Vulkan::TerrainLayout layout = w.terrain().layout(0);
        RegionId r;
        r.level                       = 0;
        r.x                           = 15;
        r.z                           = 15;
        Vulkan::TerrainTileData t     = generateTile(w.terrain().heightSource(0), layout, r);
        const auto t0                 = Clock::now();
        const Vulkan::GrassMaskData m = generateGrassMask(t, layout, std::nullopt, 0.82, 0.4);
        std::cout << "grass mask at " << spacing << " m: " << ms(t0, Clock::now()) << " ms (" << m.size << "^2)"
                  << std::endl;
    }
    // The patch step with resident tiles (recording sink): the schedule's terrain/visual time without a GPU
    World w(setupWith(integerTestHook, 1));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    TerrainViewSettings v = w.terrain().viewSettings();
    v.residencyRange0     = 600;
    w.terrain().setViewSettings(v);
    const Vec3 camera(3968, w.terrainHeight(0, 3968, 3968) + 300, 3940);
    ASSERT_TRUE(w.terrain().completeView(0, camera));
    std::array<double, 4> targets;
    for (int c = 0; c < 4; ++c) targets[size_t(c)] = w.terrainHeight(0, 3840 + 256 * (c % 2), 3840 + 256 * (c / 2));
    std::vector<double> visual, apply;
    std::vector<double> patched;
    const uint64_t hooks = w.terrain().baseHookSamples();
    for (int e = 0; e < 600; ++e)
    {
        const auto t0 = Clock::now();
        applyTerrainEdit(w, 0, scheduleBrush(e, targets));
        apply.push_back(ms(t0, Clock::now()));
        w.terrain().updateView(0, camera);
        visual.push_back(w.terrain().stats().lastVisualMs);
        patched.push_back(double(w.terrain().stats().patchedSamplesLastUpdate));
    }
    EXPECT_EQ(w.terrain().baseHookSamples(), hooks);  // gathers from resident tiles, patches from the store
    EXPECT_EQ(w.terrain().dirtyQueue().size(), 0u);
    std::cout << "resident tiles (recording sink), 600 edits:" << std::endl;
    print("applyTerrainEdit", stat(apply));
    print("terrain/visual per frame", stat(visual));
    const Stat p = stat(patched);
    std::cout << "  patched samples per frame: median " << p.median << ", p95 " << p.p95 << std::endl;
    w.setRenderSink(nullptr);
}
