/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// RenderScene's terrain (docs/engine_design.md §10.1, §12.2; S20) driven with test tiles of the pure generation and the
// integer hooks of terrain_test_hooks.h: the GPU terrain set (TerrainAtlas: uploads, the byte cap, patches, rebuilds,
// limits, memory, node bounds).

#include "saiga/vulkan/scene/GrassPass.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/TerrainAtlas.h"
#include "saiga/vulkan/scene/TerrainPass.h"
#include "saiga/vulkan/scene/TerrainTiles.h"

#include "gtest/gtest.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "render_scene_test_context.h"
#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Vulkan;
using namespace TerrainTestHooks;

namespace
{
class VulkanTerrain : public RenderSceneTest
{
   protected:
    // The tiles of every streamed level that contain the chunk (cx, cz)
    void SetPath(const TerrainLayout& layout, TerrainHeightHook hook, int32_t cx, int32_t cz, SpaceId space = 0)
    {
        for (int k = 0; k < layout.streamedLevels(); ++k)
        {
            RegionId r;
            r.space = space;
            r.level = uint8_t(k);
            r.x     = cx >> k;
            r.z     = cz >> k;
            scene->setTerrainTile(r, testTile(hook, layout, r));
        }
    }
    static RegionId Region(int level, int32_t x, int32_t z, SpaceId space = 0)
    {
        RegionId r;
        r.space = space;
        r.level = uint8_t(level);
        r.x     = x;
        r.z     = z;
        return r;
    }
};
}  // namespace

// 150 tiles at once (30 in each of the 5 levels of a 1 m layout): each frame records at most the cap, all are
// GPU-resident after the expected number of frames, the staging buffers keep their size
TEST_F(VulkanTerrain, UploadBudgetSpreadsBursts)
{
    const TerrainLayout layout = testLayout(128, 1);
    scene->setTerrainPyramid(0, testPyramid(integerTestHook, layout));
    EXPECT_EQ(renderUntilUploaded(*scene), 1);
    EXPECT_TRUE(scene->terrainAtlasStats().pyramidResident);
    const size_t staging = scene->terrainAtlasStats().memory.staging;
    for (int k = 0; k < 5; ++k)
        for (int i = 0; i < 30; ++i)
        {
            const RegionId r = Region(k, i % 6, i / 6);
            scene->setTerrainTile(r, testTile(integerTestHook, layout, r));
        }
    const size_t cap     = scene->parameters().terrainUploadBytesPerFrame;
    const size_t perTile = ((size_t(257) * 257 * 4 + 15) & ~size_t(15)) + 16;  // the layer + its page entry
    const int perFrame   = int(cap / perTile);
    const int expected   = (150 + perFrame - 1) / perFrame;
    const size_t queued  = scene->terrainUploadsPending();
    EXPECT_EQ(queued, 150 * perTile);
    int frames = 0;
    while (scene->terrainUploadsPending() > 0 && frames < 20)
    {
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
        frames++;
        const auto& s = scene->terrainAtlasStats();
        EXPECT_LE(s.bytesRecorded, cap);
        int resident = 0;
        for (int k = 0; k < 5; ++k) resident += s.tilesResident[size_t(k)];
        EXPECT_EQ(resident, std::min(150, frames * perFrame)) << "frame " << frames;
    }
    EXPECT_EQ(frames, expected);
    for (int k = 0; k < 5; ++k)
    {
        EXPECT_EQ(scene->terrainAtlasStats().tilesSet[size_t(k)], 30);
        EXPECT_EQ(scene->terrainAtlasStats().tilesResident[size_t(k)], 30);
    }
    EXPECT_EQ(scene->terrainAtlasStats().memory.staging, staging);
}

TEST_F(VulkanTerrain, UploadsMatchData)
{
    for (int spacing : {1, 2})
    {
        const TerrainLayout layout = testLayout(16, spacing);
        const auto pyramid         = testPyramid(slopeRidgeHook, layout);
        scene->setTerrainPyramid(0, pyramid);
        std::vector<RegionId> regions = {Region(0, 3, 4), Region(0, 15, 15), Region(2, 1, 3),
                                         Region(layout.streamedLevels() - 1, 0, 0)};
        std::vector<TerrainTileData> data;
        for (const RegionId& r : regions)
        {
            data.push_back(testTile(slopeRidgeHook, layout, r));
            scene->setTerrainTile(r, data.back());
        }
        renderUntilUploaded(*scene);
        for (size_t i = 0; i < regions.size(); ++i)
            EXPECT_EQ(scene->terrainAtlas().readLayerForTests(regions[i]), *data[i].heights);
        EXPECT_EQ(scene->terrainAtlas().readPyramidForTests(), *pyramid.heights);
    }
}

TEST_F(VulkanTerrain, ReplaceInPlace)
{
    const TerrainLayout layout = testLayout(16, 2);
    scene->setTerrainPyramid(0, testPyramid(flatHook, layout));
    const RegionId r = Region(1, 2, 2);
    scene->setTerrainTile(r, testTile(slopeRidgeHook, layout, r));
    renderUntilUploaded(*scene);
    const auto replacement = testTile(integerTestHook, layout, r, 9);
    scene->setTerrainTile(r, replacement);
    EXPECT_EQ(scene->terrainAtlasStats().tilesSet[1], 1);
    // Still resident with the old data until the copy is recorded
    EXPECT_EQ(scene->terrainAtlasStats().tilesResident[1], 1);
    renderUntilUploaded(*scene);
    EXPECT_EQ(scene->terrainAtlas().readLayerForTests(r), *replacement.heights);
    EXPECT_EQ(scene->terrainAtlasStats().tilesSet[1], 1);
}

TEST_F(VulkanTerrain, PatchTileAndPyramid)
{
    const TerrainLayout layout = testLayout(16, 2);
    const auto pyramid         = testPyramid(flatHook, layout);
    scene->setTerrainPyramid(0, pyramid);
    SetPath(layout, flatHook, 0, 0);
    const RegionId r = Region(0, 0, 0);
    // A patch issued while the tile's full copy is still queued lands behind it
    const size_t before = scene->terrainUploadsPending();
    const int x0 = 5, z0 = 40, w = 7, h = 3;
    std::vector<int32_t> patch(size_t(w * h));
    for (int i = 0; i < w * h; ++i) patch[size_t(i)] = 300 * 1024 + i;
    scene->patchTerrainTile(r, x0, z0, w, h, patch.data());
    // Patches count against the byte cap: 3 rows of 7 values (16-byte aligned rows)
    EXPECT_EQ(scene->terrainUploadsPending(), before + 3 * 32);
    renderUntilUploaded(*scene);
    std::vector<int32_t> expected = *testTile(flatHook, layout, r).heights;
    const int n                   = layout.samplesPerSide();
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) expected[size_t((z0 + j) * n + x0 + i)] = patch[size_t(j * w + i)];
    EXPECT_EQ(scene->terrainAtlas().readLayerForTests(r), expected);

    // The bounds of the tile's node and of every ancestor widen
    const TerrainAtlas& atlas = scene->terrainAtlas();
    for (int k = 0; k <= layout.rootLevel(); ++k)
    {
        const TerrainAtlas::Bound b = atlas.bound(k, 0, (z0 / 32) >> k);
        EXPECT_TRUE(b.valid()) << "level " << k;
        EXPECT_GE(b.hi, 300 * 1024 + w * h - 1) << "level " << k;
    }

    // The pyramid
    std::vector<int32_t> pp = {7, 8, 9, 10};
    scene->patchTerrainPyramid(0, 100, 3, 2, 2, pp.data());
    renderUntilUploaded(*scene);
    std::vector<int32_t> pexp = *pyramid.heights;
    const int P               = layout.pyramidSamplesPerSide();
    pexp[size_t(3 * P + 100)] = 7;
    pexp[size_t(3 * P + 101)] = 8;
    pexp[size_t(4 * P + 100)] = 9;
    pexp[size_t(4 * P + 101)] = 10;
    EXPECT_EQ(scene->terrainAtlas().readPyramidForTests(), pexp);
    EXPECT_EQ(scene->terrainAtlasStats().patchesApplied, 2u);

    // Ignored and counted: an unset region, another space
    scene->patchTerrainTile(Region(0, 9, 9), 0, 0, 1, 1, pp.data());
    scene->patchTerrainTile(Region(0, 0, 0, 1), 0, 0, 1, 1, pp.data());
    scene->patchTerrainPyramid(1, 0, 0, 1, 1, pp.data());
    EXPECT_EQ(scene->terrainAtlasStats().patchesIgnored, 3u);
    EXPECT_EQ(scene->terrainUploadsPending(), 0u);
}

// A pyramid of another space (or layout) drops the tiles, masks and queue; one of the same space and layout keeps them
TEST_F(VulkanTerrain, NewSetOnOtherSpaceOrLayout)
{
    const TerrainLayout layout = testLayout(16, 2);
    scene->setTerrainPyramid(0, testPyramid(slopeRidgeHook, layout));
    SetPath(layout, slopeRidgeHook, 3, 3);
    GrassMaskData mask;
    mask.size    = layout.grassMaskSize();
    mask.density = std::make_shared<std::vector<uint8_t>>(size_t(mask.size * mask.size), uint8_t(255));
    ChunkId chunk;
    chunk.x = 3;
    chunk.z = 3;
    scene->setGrassMask(chunk, mask);
    renderUntilUploaded(*scene);
    // Frames in flight read the old set
    scene->setCamera(0, Vec3(800, 100, 800), Quat::Identity(), 1.0);
    scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    const uint64_t rebuilds = scene->terrainAtlasStats().rebuilds;

    // The same space and layout: kept
    scene->setTerrainPyramid(0, testPyramid(integerTestHook, layout));
    EXPECT_EQ(scene->terrainAtlasStats().rebuilds, rebuilds);
    EXPECT_EQ(scene->terrainAtlasStats().tilesSet[0], 1);
    EXPECT_EQ(scene->terrainAtlasStats().masksSet, 1);
    scene->setTerrainTile(Region(0, 4, 4), testTile(slopeRidgeHook, layout, Region(0, 4, 4)));

    // Space 1: a new set (the queued tile is dropped too)
    const auto other = testPyramid(flatHook, layout);
    scene->setTerrainPyramid(1, other);
    const auto& s = scene->terrainAtlasStats();
    EXPECT_EQ(s.rebuilds, rebuilds + 1);
    EXPECT_EQ(s.space, 1);
    EXPECT_EQ(s.tilesSet[0], 0);
    EXPECT_EQ(s.masksSet, 0);
    EXPECT_FALSE(s.pyramidResident);
    const size_t P = size_t(layout.pyramidSamplesPerSide());
    EXPECT_EQ(scene->terrainUploadsPending(), (P * P * 4 + 15) & ~size_t(15));
    for (int i = 0; i < 4; ++i) scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    EXPECT_TRUE(scene->terrainAtlasStats().pyramidResident);
    EXPECT_EQ(scene->terrainAtlas().readPyramidForTests(), *other.heights);

    // Another layout of the same space: a new set
    scene->setTerrainPyramid(1, testPyramid(flatHook, testLayout(16, 1)));
    EXPECT_EQ(scene->terrainAtlasStats().rebuilds, rebuilds + 2);
    renderUntilUploaded(*scene);

    // S21, the switches of the camera between spaces with different layouts: space 0 at 1 m, space 2 at 2 m, space 0
    // again, then an enclosed camera space (clearTerrain); frames in flight on every switch (validation clean)
    const Vec3 camera(800, 120, 900);
    const Quat down = Quat(std::cos(-0.35), std::sin(-0.35), 0, 0);  // 40 degrees down
    auto frames     = [&](SpaceId space, int n)
    {
        scene->setCamera(space, camera, down, 1.0);
        for (int i = 0; i < n; ++i) scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
        scene->waitIdle();
        return scene->terrainPassStats();
    };
    const TerrainLayout fine = testLayout(16, 1), coarse = testLayout(16, 2);
    fillTestTerrain(*scene, fine, TerrainHeightSource::fromHook(integerTestHook, 1), std::nullopt, camera, false, 0);
    const TerrainPass::Stats first = frames(0, 3);
    EXPECT_GT(first.nodes, 0);
    EXPECT_EQ(scene->terrainAtlasStats().space, 0);
    fillTestTerrain(*scene, coarse, TerrainHeightSource::fromHook(slopeRidgeHook, 1), std::nullopt, camera, false, 2);
    const TerrainPass::Stats second = frames(2, 3);
    EXPECT_GT(second.nodes, 0);
    EXPECT_EQ(scene->terrainAtlasStats().space, 2);
    EXPECT_EQ(scene->terrainAtlasStats().layout.baseSpacing, 2);
    fillTestTerrain(*scene, fine, TerrainHeightSource::fromHook(integerTestHook, 1), std::nullopt, camera, false, 0);
    const TerrainPass::Stats third = frames(0, 3);
    EXPECT_EQ(scene->terrainAtlasStats().layout.baseSpacing, 1);
    // The same layout and camera: the same selection per level as the first time
    EXPECT_EQ(third.nodesPerLevel, first.nodesPerLevel);
    EXPECT_EQ(third.quartersPerLevel, first.quartersPerLevel);
    scene->clearTerrain();
    const TerrainPass::Stats none = frames(1, 3);
    EXPECT_EQ(none.nodes, 0);
    EXPECT_FALSE(scene->terrainAtlasStats().hasSet);
}

TEST_F(VulkanTerrain, ClearTerrain)
{
    const TerrainLayout layout = testLayout(16, 2);
    scene->setTerrainPyramid(0, testPyramid(slopeRidgeHook, layout));
    SetPath(layout, slopeRidgeHook, 1, 1);
    renderUntilUploaded(*scene);
    scene->setCamera(0, Vec3(300, 100, 300), Quat::Identity(), 1.0);
    scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    const uint64_t rebuilds = scene->terrainAtlasStats().rebuilds;
    scene->clearTerrain();
    EXPECT_FALSE(scene->terrainAtlasStats().hasSet);
    EXPECT_EQ(scene->terrainUploadsPending(), 0u);
    for (int i = 0; i < 3; ++i) scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    scene->setTerrainPyramid(0, testPyramid(slopeRidgeHook, layout));
    EXPECT_TRUE(scene->terrainAtlasStats().hasSet);
    EXPECT_EQ(scene->terrainAtlasStats().rebuilds, rebuilds + 1);
    renderUntilUploaded(*scene);
}

TEST_F(VulkanTerrain, AtlasLimits)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // A 1 m layout of 128 chunks needs a 63 MB atlas: more than a 32 MiB storage-buffer limit (every build)
    EXPECT_DEATH(
        {
            RenderSceneParameters p;
            p.storageBufferLimit = 32u << 20;
            Create(64, 48, 1, p);
            scene->setTerrainPyramid(0, testPyramid(flatHook, testLayout(128, 1)));
        },
        "storage buffer limit");
#ifdef SAIGA_ASSERTS
    // A new tile on a full level
    EXPECT_DEATH(
        {
            TerrainLayout layout = testLayout(16, 2);
            layout.tilesPerLevel = 2;
            scene->setTerrainPyramid(0, testPyramid(flatHook, layout));
            for (int i = 0; i < 3; ++i)
                scene->setTerrainTile(Region(0, i, 0), testTile(flatHook, layout, Region(0, i, 0)));
        },
        "no free layer");
#endif
}

TEST_F(VulkanTerrain, MemoryMatchesLayout)
{
    for (const TerrainLayout& layout : {testLayout(16, 2), testLayout(32, 1), testLayout(128, 1)})
    {
        scene->setTerrainPyramid(0, testPyramid(flatHook, layout));
        const TerrainMemory est = TerrainAtlas::memoryEstimate(layout, scene->parameters());
        const TerrainMemory m   = scene->terrainAtlasStats().memory;
        EXPECT_EQ(m.atlas, est.atlas);
        EXPECT_EQ(m.pyramid, est.pyramid);
        EXPECT_EQ(m.pageTable, est.pageTable);
        EXPECT_EQ(m.masks, est.masks);
        EXPECT_EQ(m.nodes, est.nodes);
        EXPECT_EQ(m.staging, est.staging);
        EXPECT_EQ(m.bounds, est.bounds);
        EXPECT_EQ(m.gpuBytes(), est.gpuBytes());
        EXPECT_EQ(m.cpuBytes(), est.cpuBytes());
        // The defaults equal the estimate's defaults
        EXPECT_EQ(TerrainAtlas::memoryEstimate(layout).gpuBytes(), est.gpuBytes());
    }
    // The 32 km layout at 1 m: about 63 MB of atlas, 22 MB of node bounds
    const TerrainMemory big = TerrainAtlas::memoryEstimate(testLayout(128, 1));
    EXPECT_EQ(big.atlas, size_t(240) * 257 * 257 * 4);
    EXPECT_EQ(big.bounds, size_t(1398080) * 16);
    renderUntilUploaded(*scene);
}

// The CPU bound of every ancestor contains a resident level-0 tile's 300 m spike; removing the tile shrinks them again
TEST_F(VulkanTerrain, BoundsIncludeFinerTilesCpu)
{
    const TerrainLayout layout = testLayout(32, 1);
    // At odd metres: only level 0 (1 m) samples the spike
    spikeTable() = {Spike{301, 521, 300 * 1024}};
    scene->setTerrainPyramid(0, testPyramid(spikeHook, layout));
    // The pyramid's 32 m samples miss the spike: flat bounds
    const int R = layout.rootLevel();
    EXPECT_EQ(scene->terrainAtlas().bound(R, 0, 0).hi, 0);
    // The path to chunk (1, 2) (the spike's chunk)
    SetPath(layout, spikeHook, 1, 2);
    const TerrainAtlas& atlas = scene->terrainAtlas();
    for (int k = 0; k <= R; ++k)
    {
        // The node of level k containing (301, 521): size 32 << k
        const int32_t nx = 301 / (32 << k), nz = 521 / (32 << k);
        EXPECT_EQ(atlas.bound(k, nx, nz).hi, 300 * 1024) << "level " << k;
        if (k > 0)
        {
            EXPECT_EQ(atlas.ownBound(k, nx, nz).hi, 0) << "level " << k;
        }
    }
    scene->removeTerrainTile(Region(0, 1, 2));
    for (int k = 1; k <= R; ++k) EXPECT_EQ(atlas.bound(k, 301 / (32 << k), 521 / (32 << k)).hi, 0) << "level " << k;
    EXPECT_FALSE(atlas.bound(0, 301 / 32, 521 / 32).valid());
    spikeTable().clear();
    renderUntilUploaded(*scene);
}

// Regression (PR review): a replaced GPU-resident tile or pyramid had only the new data's bounds while the GPU still
// held the old samples (until the tile's copy, while the pyramid's rows are recorded), so culling could reject terrain
// drawn at the old heights. A replacement of data not yet on the GPU takes the new bounds
TEST_F(VulkanTerrain, ReplacedResidentDataKeepsBounds)
{
    const TerrainLayout layout = testLayout(32, 1);
    // (301, 521): only level 0 samples it; (320, 512): a pyramid sample
    spikeTable()              = {Spike{301, 521, 300 * 1024}, Spike{320, 512, 200 * 1024}};
    const TerrainAtlas& atlas = scene->terrainAtlas();
    const RegionId r          = Region(0, 1, 2);
    auto tileHi               = [&]() { return atlas.ownBound(0, 301 / 32, 521 / 32).hi; };
    // The finest pyramid level (32 m samples)
    const int P    = layout.streamedLevels();
    auto pyramidHi = [&]() { return atlas.ownBound(P, 320 / (32 << P), 512 / (32 << P)).hi; };
    scene->setTerrainPyramid(0, testPyramid(flatHook, layout));
    scene->setTerrainPyramid(0, testPyramid(spikeHook, layout));
    EXPECT_EQ(pyramidHi(), 200 * 1024);
    scene->setTerrainPyramid(0, testPyramid(flatHook, layout));
    EXPECT_EQ(pyramidHi(), 0) << "the replaced pyramid was never GPU-resident";
    scene->setTerrainTile(r, testTile(spikeHook, layout, r));
    scene->setTerrainTile(r, testTile(flatHook, layout, r));
    EXPECT_EQ(tileHi(), 0) << "the replaced tile was never GPU-resident";

    scene->setTerrainPyramid(0, testPyramid(spikeHook, layout));
    scene->setTerrainTile(r, testTile(spikeHook, layout, r));
    renderUntilUploaded(*scene);
    ASSERT_TRUE(atlas.pyramidResident());
    ASSERT_TRUE(atlas.tileResident(0, 1, 2));
    scene->setTerrainPyramid(0, testPyramid(flatHook, layout));
    scene->setTerrainTile(r, testTile(flatHook, layout, r));
    EXPECT_EQ(pyramidHi(), 200 * 1024);
    EXPECT_EQ(tileHi(), 300 * 1024);
    // Regression (PR review): a negative level reached a negative shift count
    EXPECT_FALSE(atlas.tileSet(-1, 0, 0));
    EXPECT_FALSE(atlas.tileResident(-1, 0, 0));
    spikeTable().clear();
    renderUntilUploaded(*scene);
}

// ---------------------------------------------------------------------------------------------------------------------
// Step 4b: the terrain pass (TerrainPass)

namespace
{
// The camera rotation looking along the direction (y up)
Quat lookRotation(const Vec3& forward)
{
    const Vec3 f = forward.normalized();
    const Vec3 r = f.cross(Vec3(0, 1, 0)).normalized();
    const Vec3 u = r.cross(f);
    Mat3 m;
    for (int i = 0; i < 3; ++i)
    {
        m(i, 0) = r(i);
        m(i, 1) = u(i);
        m(i, 2) = -f(i);
    }
    return Quat(m).normalized();
}
// Along yaw (degrees from +x towards +z) and pitch (degrees, negative: down)
Vec3 direction(double yawDeg, double pitchDeg)
{
    const double y = yawDeg * kPi / 180, p = pitchDeg * kPi / 180;
    return Vec3(std::cos(p) * std::cos(y), std::sin(p), std::cos(p) * std::sin(y));
}

// The CPU surface of a hook at spacing bs (the §12.2 rule over the clamped samples), metres
double surfaceAt(TerrainHeightHook hook, uint64_t seed, int bs, double x, double z)
{
    const auto src  = TerrainHeightSource::fromHook(hook, seed);
    const double gx = x / bs, gz = z / bs;
    const int64_t i = int64_t(std::floor(gx)), j = int64_t(std::floor(gz));
    const double fx = gx - double(i), fz = gz - double(j);
    return terrainSurfaceDouble(src(i * bs, j * bs), src((i + 1) * bs, j * bs), src(i * bs, (j + 1) * bs),
                                src((i + 1) * bs, (j + 1) * bs), fx, fz) *
           kHeightUnit;
}

int countColor(const TemplatedImage<ucvec4>& img, const ucvec4& c, int tolerance = 8)
{
    int n = 0;
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x)
        {
            bool same = true;
            for (int k = 0; k < 3; ++k) same = same && std::abs(int(img(y, x)(k)) - int(c(k))) <= tolerance;
            n += same ? 1 : 0;
        }
    return n;
}

const ucvec4 kClear(57, 57, 57, 255);
}  // namespace

// §22 acceptance (S20): level 0 has one vertex per sample, at the sample's height (both spacings)
TEST_F(VulkanTerrain, OneVertexPerSampleAtLevelZero)
{
    for (int bs : {1, 2})
    {
        const TerrainLayout layout = testLayout(16, bs);
        const Vec3 eye(1100.3, 0, 1300.7);
        const double ground = surfaceAt(slopeRidgeHook, 1, bs, eye.x(), eye.z());
        const Vec3 cam      = eye + Vec3(0, ground + 30, 0);
        fillTestTerrain(*scene, layout, TerrainHeightSource::fromHook(slopeRidgeHook, 1), std::nullopt, cam, false);
        SetCamera(0, cam, lookRotation(direction(20, -45)), 1.0);
        Capture(2);
        const auto& nodes = scene->terrainPass().selectedNodes();
        std::vector<TerrainProbe> probes;
        std::vector<Vec3> expected;
        for (size_t n = 0; n < nodes.size(); ++n)
        {
            if (nodes[n].level != 0 || nodes[n].quarter || nodes[n].water) continue;
            for (int g = 0; g <= 32; g += 3)
            {
                const int gx = g, gz = (g * 7) % 33;
                const Vec3 p(double(nodes[n].originX + gx * bs), 0, double(nodes[n].originZ + gz * bs));
                // Unmorphed: within the morph start (0.7 * 300 m)
                if ((p - cam).norm() > 150) continue;
                const double h =
                    double(clampTerrainHeight(slopeRidgeHook(1, int64_t(p.x()), int64_t(p.z())))) * kHeightUnit;
                probes.push_back(TerrainProbe{int(n), gx, gz});
                expected.push_back(Vec3(p.x(), h, p.z()));
            }
        }
        ASSERT_GT(probes.size(), 50u) << "spacing " << bs;
        const auto out = scene->terrainPass().probeVertices(probes);
        for (size_t i = 0; i < probes.size(); ++i)
        {
            const Vec3 world = cam + Vec3(out[i](0), out[i](1), out[i](2));
            const double d   = (expected[i] - cam).norm();
            // xz on the sample grid (exact up to the one float rounding of the camera-relative value)
            EXPECT_NEAR(world.x(), expected[i].x(), 1e-6 * d + 1e-4) << "spacing " << bs;
            EXPECT_NEAR(world.z(), expected[i].z(), 1e-6 * d + 1e-4) << "spacing " << bs;
            EXPECT_NEAR(world.y(), expected[i].y(), 1e-6 * d + 1e-4) << "spacing " << bs;
        }
    }
}

// Partly morphed vertices lie at mix(h(i, j), h(i & ~1, j & ~1), k); the edge vertices of two adjacent nodes of the
// same level are bit-identical
TEST_F(VulkanTerrain, MorphTargetsAndSharedVertices)
{
    const int bs               = 1;
    const TerrainLayout layout = testLayout(16, bs);
    const Vec3 cam(1500.5, 120, 1500.5);
    fillTestTerrain(*scene, layout, TerrainHeightSource::fromHook(highFrequencyHook, 1), std::nullopt, cam, false);
    SetCamera(0, cam, lookRotation(direction(45, -30)), 1.2);
    Capture(2);
    const auto& nodes  = scene->terrainPass().selectedNodes();
    const auto& frameU = scene->terrainPass().frameUniforms();
    const double range = scene->terrainPass().usedRange0();
    const double start = 0.7 * range;
    // The LOD metric: the horizontal distance with the frame's camera height above the surface under it
    const double hc = scene->terrainPass().usedLodHeight();
    EXPECT_NEAR(hc, cam.y() - surfaceAt(highFrequencyHook, 1, bs, cam.x(), cam.z()), 6.0);
    const Vec3 crel(frameU.cameraRel(0), frameU.cameraRel(1), frameU.cameraRel(2));
    const int32_t ref = frameU.anchor(2);
    std::vector<TerrainProbe> probes;
    std::vector<Vec3> expected;
    std::vector<double> ks;
    int morphed = 0;
    for (size_t n = 0; n < nodes.size(); ++n)
    {
        if (nodes[n].level != 0 || nodes[n].quarter) continue;
        for (int gz = 0; gz <= 32; ++gz)
            for (int gx = 1; gx <= 32; gx += 2)
            {
                const int64_t x = nodes[n].originX + gx, z = nodes[n].originZ + gz;
                const int64_t xt = nodes[n].originX + (gx & ~1), zt = nodes[n].originZ + (gz & ~1);
                const int32_t h = highFrequencyHook(1, x, z), ht = highFrequencyHook(1, xt, zt);
                const double dist = Vec3(double(x) - cam.x(), hc, double(z) - cam.z()).norm();
                const double k    = std::min(1.0, std::max(0.0, (dist - start) / (0.99 * range - start)));
                if (k <= 0.05 || k >= 0.95) continue;
                morphed++;
                probes.push_back(TerrainProbe{int(n), gx, gz});
                const double mx = double(x) + (double(xt) - double(x)) * k,
                             mz = double(z) + (double(zt) - double(z)) * k;
                const double my = (double(h - ref) + double(ht - h) * k) * kHeightUnit - crel.y() + cam.y();
                expected.push_back(Vec3(mx, my, mz));
                ks.push_back(k);
            }
    }
    ASSERT_GT(morphed, 100);
    const auto out = scene->terrainPass().probeVertices(probes);
    for (size_t i = 0; i < probes.size(); ++i)
    {
        EXPECT_NEAR(out[i](3), ks[i], 1e-4);
        const Vec3 world = cam + Vec3(out[i](0), out[i](1), out[i](2));
        // k from the float distance: a few 1e-4 m
        EXPECT_NEAR(world.x(), expected[i].x(), 2e-3);
        EXPECT_NEAR(world.z(), expected[i].z(), 2e-3);
        EXPECT_NEAR(world.y(), expected[i].y(), 2e-2);
    }
    // Shared edges: a node's x = 32 column against its +x neighbour's x = 0 column (same level, same integers)
    int pairs = 0;
    for (size_t a = 0; a < nodes.size(); ++a)
        for (size_t b = 0; b < nodes.size(); ++b)
        {
            if (nodes[a].level != nodes[b].level || nodes[a].quarter || nodes[b].quarter || nodes[a].water ||
                nodes[b].water)
                continue;
            const int size = 32 << nodes[a].level;
            if (nodes[b].originX != nodes[a].originX + size || nodes[b].originZ != nodes[a].originZ) continue;
            std::vector<TerrainProbe> edge;
            for (int gz = 0; gz <= 32; ++gz)
            {
                edge.push_back(TerrainProbe{int(a), 32, gz});
                edge.push_back(TerrainProbe{int(b), 0, gz});
            }
            const auto v = scene->terrainPass().probeVertices(edge);
            for (size_t i = 0; i < v.size(); i += 2)
                for (int c = 0; c < 3; ++c) EXPECT_EQ(v[i](c), v[i + 1](c));
            pairs++;
        }
    EXPECT_GT(pairs, 5);
}

// The real vertex pipeline: an oblique view of a level-0 area; the rendered depth unprojected at pixel centres lies
// on the CPU surface rule
TEST_F(VulkanTerrain, RenderedDepthMatchesSurface)
{
    Create(96, 64, 1);
    const int bs               = 1;
    const TerrainLayout layout = testLayout(16, bs);
    const Vec3 eye(2000.25, 0, 1800.75);
    const Vec3 cam = eye + Vec3(0, surfaceAt(slopeRidgeHook, 1, bs, eye.x(), eye.z()) + 20, 0);
    fillTestTerrain(*scene, layout, TerrainHeightSource::fromHook(slopeRidgeHook, 1), std::nullopt, cam, false);
    SetCamera(0, cam, lookRotation(direction(10, -40)), 60.0 * kPi / 180);
    Capture(2);
    int w = 0, h = 0;
    const std::vector<float> depth = scene->readDepthForTests(w, h);
    ASSERT_EQ(w, 96);
    const Mat4 vp    = scene->viewProjection().cast<double>();
    const Mat4 invVp = vp.inverse();
    int checked      = 0;
    for (int k = 0; k < 50; ++k)
    {
        const int px = 5 + (k * 37) % 86, py = 5 + (k * 23) % 54;
        const double d = depth[size_t(py) * size_t(w) + size_t(px)];
        ASSERT_GT(d, 0.0) << "sky at " << px << ", " << py;
        const Vec4 ndc((px + 0.5) / w * 2 - 1, (py + 0.5) / h * 2 - 1, d, 1);
        Vec4 r = invVp * ndc;
        r /= r(3);
        const Vec3 p      = cam + Vec3(r(0), r(1), r(2));
        const double dist = Vec3(r(0), r(1), r(2)).norm();
        EXPECT_NEAR(p.y(), surfaceAt(slopeRidgeHook, 1, bs, p.x(), p.z()), 1e-3 * dist) << px << ", " << py;
        checked++;
    }
    EXPECT_EQ(checked, 50);
}

// §22 acceptance (S20, A12): at 28 672 m a 1 mm camera move moves no vertex by more than 0.5 px against the double
// projection (probes from 0.5 m to 50 m at levels 0 to 3, a 0.25 mm/px footprint at the nearest probe); a
// float-absolute camera would be off by more than 4 px
TEST_F(VulkanTerrain, CameraMoveAt30km)
{
    const int bs               = 1;
    const TerrainLayout layout = testLayout(128, bs);
    const auto source          = TerrainHeightSource::fromHook(slopeRidgeHook, 1);
    // x and z about half a float ulp (2^-9 m at 28 km) off the float grid in opposite directions, the view along
    // (1, 0, 1): the float-absolute camera's error is lateral (about 1.35 mm)
    const Vec3 eye(28682 + 189.49 / 512, 0, 28677 + 365.51 / 512);
    const double ground = surfaceAt(slopeRidgeHook, 1, bs, eye.x(), eye.z());
    const auto pyramid  = generateTerrainPyramid(source, layout, std::nullopt);
    double worstMove = 0, worstBug = 0;
    for (int finest = 0; finest <= 3; ++finest)
    {
        // Only levels >= finest are resident: the nodes around the camera are of that level
        const Vec3 cam0 = eye + Vec3(0, ground + 0.3, 0);
        scene->setTerrainPyramid(0, pyramid);
        for (int k = 0; k < layout.streamedLevels(); ++k)
            for (const RegionId& r : desiredTestTiles(layout, pyramid, cam0, 0, k))
                if (r.level == k)
                {
                    if (k >= finest)
                        scene->setTerrainTile(r, generateTerrainTile(source, layout, r));
                    else
                        scene->removeTerrainTile(r);
                }
        renderUntilUploaded(*scene, 256);
        const Quat rot = lookRotation(direction(45, -20));
        const Mat3 Rt  = rot.matrix().transpose();
        // The focal length (px per unit of x / -z) of a 0.25 mm/px footprint at the nearest probe (set per frame)
        double f     = 2000;
        auto project = [&](const Vec3& rel)
        {
            const Vec3 v = Rt * rel;
            return Vec2(f * v.x() / -v.z(), f * v.y() / -v.z());
        };
        for (int move = 0; move < 2; ++move)
        {
            const Vec3 cam = cam0 + Vec3(move * 0.001, 0, 0);
            SetCamera(0, cam, rot, 1.0);
            Capture(2);
            const auto& nodes = scene->terrainPass().selectedNodes();
            std::vector<TerrainProbe> probes;
            std::vector<Vec3> truth;
            for (size_t n = 0; n < nodes.size(); ++n)
            {
                if (nodes[n].level != finest || nodes[n].water) continue;
                const int sp = bs << finest, grid = nodes[n].quarter ? 16 : 32;
                for (int gz = 0; gz <= grid; ++gz)
                    for (int gx = 0; gx <= grid; ++gx)
                    {
                        // Within 50 m every level's morph factor is 0 (the morph starts beyond 0.7 * 300 m)
                        const int64_t x = nodes[n].originX + gx * sp, z = nodes[n].originZ + gz * sp;
                        const Vec3 p(double(x), double(source(x, z)) * kHeightUnit, double(z));
                        const Vec3 rel = p - cam;
                        const double d = rel.norm();
                        // In the view (within 30 degrees of the view axis, as in an image)
                        const Vec3 v = Rt * rel;
                        if (d < 0.5 || d > 50 || v.z() > -0.1 || std::abs(v.x() / v.z()) > 0.58 ||
                            std::abs(v.y() / v.z()) > 0.58)
                            continue;
                        probes.push_back(TerrainProbe{int(n), gx, gz});
                        truth.push_back(p);
                    }
            }
            ASSERT_GT(probes.size(), 10u) << "level " << finest;
            double nearest = 1e9;
            for (const Vec3& t : truth) nearest = std::min(nearest, (t - cam).norm());
            f              = nearest / 0.00025;
            const auto out = scene->terrainPass().probeVertices(probes);
            for (size_t i = 0; i < probes.size(); ++i)
            {
                const Vec3 gpu(out[i](0), out[i](1), out[i](2));
                const double e = (project(gpu) - project(truth[i] - cam)).norm();
                EXPECT_LE(e, 0.25) << "level " << finest << " move " << move << " distance " << (truth[i] - cam).norm();
                worstMove = std::max(worstMove, e);
                // The bug: float-absolute positions minus a float camera
                const Vec3 bug(double(float(truth[i].x()) - float(cam.x())),
                               double(float(truth[i].y()) - float(cam.y())),
                               double(float(truth[i].z()) - float(cam.z())));
                worstBug = std::max(worstBug, (project(bug) - project(truth[i] - cam)).norm());
            }
        }
    }
    std::cout << "CameraMoveAt30km: worst error " << worstMove << " px, float-absolute camera " << worstBug << " px"
              << std::endl;
    EXPECT_GT(worstBug, 4.0);
}

// §22 acceptance (S20): no background pixel between a level-0 tile and an area only level 1 covers (high-frequency
// heights); without skirts the view sees cracks
TEST_F(VulkanTerrain, NoCrackAtLevelBorders)
{
    RenderSceneParameters p;
    p.clearColor = vec4(1, 0, 1, 1);
    Create(320, 200, 4, p);
    const int bs               = 1;
    const TerrainLayout layout = testLayout(16, bs);
    const auto source          = TerrainHeightSource::fromHook(highFrequencyHook, 1);
    const auto pyramid         = generateTerrainPyramid(source, layout, std::nullopt);
    scene->setTerrainPyramid(0, pyramid);
    // Every level from 1 around the border; level 0 only for chunk (2, 2): the border at x = 768
    const Vec3 cam(738, 0, 640);
    for (const RegionId& r : desiredTestTiles(layout, pyramid, cam, 0, 1))
        scene->setTerrainTile(r, generateTerrainTile(source, layout, r));
    const RegionId level0 = Region(0, 2, 2);
    scene->setTerrainTile(level0, generateTerrainTile(source, layout, level0));
    renderUntilUploaded(*scene, 256);
    const Vec3 eye = cam + Vec3(0, surfaceAt(highFrequencyHook, 1, bs, cam.x(), cam.z()) + 40, 0);
    // Across the border 35 degrees down, the border off-centre (yawed)
    SetCamera(0, eye, lookRotation(direction(12, -35)), 40.0 * kPi / 180);
    PrototypeSettings s = scene->prototypeSettings();
    s.terrainSkirts     = true;
    scene->setPrototypeSettings(s);
    const auto img = Capture(3);
    EXPECT_GT(scene->terrainPassStats().nodesPerLevel[0], 0);
    EXPECT_GT(scene->terrainPassStats().nodesPerLevel[1], 0);
    EXPECT_EQ(countColor(img, ucvec4(255, 0, 255, 255), 0), 0);
    s.terrainSkirts = false;
    scene->setPrototypeSettings(s);
    const auto cracks = Capture(3);
    EXPECT_GT(countColor(cracks, ucvec4(255, 0, 255, 255), 0), 10);
}

namespace
{
// A 900 m cliff facing -z: a ramp of 15 m per m from z = 2048 to 2108, with 8 times highFrequencyHook's noise (48 m)
// everywhere
int32_t cliffHook(uint64_t seed, int64_t x, int64_t z)
{
    const int64_t t = std::min<int64_t>(std::max<int64_t>(z - 2048, 0), 60);
    return int32_t(t * 15 * 1024 + 8 * (highFrequencyHook(seed, x, z) - 20 * 1024) + 20 * 1024);
}
}  // namespace

// Regression (implementation review of S20): at steep terrain the coarse side of a level border morphed its edge
// (the LOD distance was 3D: the selection's box, lowered by the skirt, was near while the cliff's edge vertices were
// far), opening gaps that the skirts filled and flat shading lit as walls (a saw-tooth). With every desired tile
// resident, at the foot of a cliff: every coarse vertex on a level border has morph factor 0 and every fine one 1 (the
// old metric: about 90 coarse vertices with factors up to 0.6), and the skirts cover no gap (the same background with
// and without them)
TEST_F(VulkanTerrain, NoGapAtLevelBordersOfACliff)
{
    RenderSceneParameters p;
    p.clearColor = vec4(1, 0, 1, 1);
    Create(320, 200, 4, p);
    const int bs         = 1;
    TerrainLayout layout = testLayout(16, bs);
    layout.tilesPerLevel = 64;
    const auto source    = TerrainHeightSource::fromHook(cliffHook, 1);
    scene->setTerrainPyramid(0, generateTerrainPyramid(source, layout, std::nullopt));
    // 100 m in front of the cliff's foot, 2 m above the ground
    const Vec3 foot(1800.5, 0, 1948.5);
    const Vec3 eye = foot + Vec3(0, surfaceAt(cliffHook, 1, bs, foot.x(), foot.z()) + 2, 0);
    // Every tile within 400 * 2^k m horizontally (the selection never refuses a refinement)
    for (int k = 0; k < layout.streamedLevels(); ++k)
    {
        const double size = double(256 << k), range = 400.0 * double(1 << k);
        int count = 0;
        for (int32_t tz = 0; tz < layout.tilesPerSide(k); ++tz)
            for (int32_t tx = 0; tx < layout.tilesPerSide(k); ++tx)
            {
                const double dx = std::max({tx * size - eye.x(), 0.0, eye.x() - (tx + 1) * size});
                const double dz = std::max({tz * size - eye.z(), 0.0, eye.z() - (tz + 1) * size});
                if (dx * dx + dz * dz > range * range) continue;
                scene->setTerrainTile(Region(k, tx, tz), generateTerrainTile(source, layout, Region(k, tx, tz)));
                count++;
            }
        ASSERT_LE(count, layout.tilesPerLevel) << "level " << k;
    }
    renderUntilUploaded(*scene, 512);
    PrototypeSettings s = scene->prototypeSettings();
    s.terrainSkirts     = false;
    s.sky               = false;
    s.water             = false;
    scene->setPrototypeSettings(s);
    // Along the cliff (+x) and up at the face where the level-0/1 border crosses it, about 300 m away
    SetCamera(0, eye, lookRotation(Vec3(283, 400, 100)), 20.0 * kPi / 180);
    const auto img = Capture(3);
    const auto& st = scene->terrainPassStats();
    EXPECT_GT(st.nodesPerLevel[0], 0);
    EXPECT_GT(st.nodesPerLevel[1], 0);
    EXPECT_EQ(st.refusedResidency, 0);
    // Gaps show the background with the skirts off and are covered with them; the sky above the cliff is background in
    // both images
    const int magentaOff = countColor(img, ucvec4(255, 0, 255, 255), 0);
    s.terrainSkirts      = true;
    scene->setPrototypeSettings(s);
    const int magentaOn = countColor(Capture(3), ucvec4(255, 0, 255, 255), 0);
    EXPECT_LT(magentaOn, width * height / 2);
    EXPECT_EQ(magentaOff, magentaOn);

    // The edge vertices of every node that lie on a node of the next finer (coarser) level
    const auto& nodes = scene->terrainPass().selectedNodes();
    auto extent       = [&](const TerrainNodeInfo& n)
    {
        const int grid = n.quarter ? 16 : 32;
        return int32_t(grid * (bs << n.level));
    };
    auto inside = [&](const TerrainNodeInfo& n, int64_t x, int64_t z)
    { return x >= n.originX && x <= n.originX + extent(n) && z >= n.originZ && z <= n.originZ + extent(n); };
    std::vector<TerrainProbe> probes;
    std::vector<float> expected;
    for (size_t a = 0; a < nodes.size(); ++a)
    {
        if (nodes[a].water) continue;
        const int grid = nodes[a].quarter ? 16 : 32, sp = bs << nodes[a].level;
        for (int gz = 0; gz <= grid; ++gz)
            for (int gx = 0; gx <= grid; ++gx)
            {
                if (gx != 0 && gz != 0 && gx != grid && gz != grid) continue;
                const int64_t x = nodes[a].originX + gx * sp, z = nodes[a].originZ + gz * sp;
                for (size_t b = 0; b < nodes.size(); ++b)
                {
                    if (b == a || nodes[b].water || std::abs(nodes[b].level - nodes[a].level) != 1) continue;
                    if (!inside(nodes[b], x, z)) continue;
                    probes.push_back(TerrainProbe{int(a), gx, gz});
                    // The coarse side unmorphed, the fine side fully morphed
                    expected.push_back(nodes[a].level > nodes[b].level ? 0.0f : 1.0f);
                    break;
                }
            }
    }
    ASSERT_GT(probes.size(), 50u);
    const auto out = scene->terrainPass().probeVertices(probes);
    int wrong      = 0;
    for (size_t i = 0; i < probes.size(); ++i) wrong += out[i](3) == expected[i] ? 0 : 1;
    EXPECT_EQ(wrong, 0) << "of " << probes.size() << " border vertices";
}

// A resident level-0 tile with a 300 m plateau under flat coarse levels: the camera looks up, so the coarse boxes alone
// are below the frustum; the hierarchical bounds keep the plateau's ancestors
TEST_F(VulkanTerrain, BoundsIncludeFinerTiles)
{
    const int bs               = 1;
    const TerrainLayout layout = testLayout(16, bs);
    spikeTable().clear();
    for (int z = 300; z <= 330; ++z)
        for (int x = 300; x <= 330; ++x) spikeTable().push_back(Spike{x, z, 300 * 1024});
    scene->setTerrainPyramid(0, testPyramid(flatHook, layout));
    const Vec3 cam(150, 10, 315);
    for (int k = 1; k < layout.streamedLevels(); ++k)
        scene->setTerrainTile(Region(k, 0, 0), testTile(flatHook, layout, Region(k, 0, 0)));
    SetCamera(0, cam, lookRotation(direction(0, 40)), 40.0 * kPi / 180);
    renderUntilUploaded(*scene);
    // Without the level-0 tile nothing is in view
    EXPECT_EQ(countColor(Capture(2), kClear, 0), width * height);
    scene->setTerrainTile(Region(0, 1, 1), testTile(spikeHook, layout, Region(0, 1, 1)));
    renderUntilUploaded(*scene);
    const int drawn = width * height - countColor(Capture(2), kClear, 0);
    EXPECT_GT(drawn, 20);
    EXPECT_GT(scene->terrainPassStats().nodesPerLevel[0], 0);
    spikeTable().clear();
}

// A node is refined only into GPU-resident tiles; a tile queued but not yet recorded is not refined
TEST_F(VulkanTerrain, RefinesOnlyResidentTiles)
{
    RenderSceneParameters p;
    // Two 1 m tiles per frame
    p.terrainUploadBytesPerFrame = 2 * (((size_t(257) * 257 * 4 + 15) & ~size_t(15)) + 16);
    Create(96, 64, 4, p);
    const int bs               = 1;
    const TerrainLayout layout = testLayout(16, bs);
    const auto source          = TerrainHeightSource::fromHook(slopeRidgeHook, 1);
    const auto pyramid         = generateTerrainPyramid(source, layout, std::nullopt);
    scene->setTerrainPyramid(0, pyramid);
    const Vec3 cam(640, surfaceAt(slopeRidgeHook, 1, bs, 640, 640) + 30, 640);
    for (const RegionId& r : desiredTestTiles(layout, pyramid, cam, 0, 1))
        scene->setTerrainTile(r, generateTerrainTile(source, layout, r));
    renderUntilUploaded(*scene, 256);
    SetCamera(0, cam, lookRotation(direction(0, -50)), 1.0);
    Capture(2);
    EXPECT_EQ(scene->terrainPassStats().nodesPerLevel[0], 0);
    EXPECT_GT(scene->terrainPassStats().nodesPerLevel[1], 0);
    EXPECT_GT(scene->terrainPassStats().refusedResidency, 0);
    // Two far tiles first, then the camera's: the first frame records only the far ones
    scene->setTerrainTile(Region(0, 10, 10), generateTerrainTile(source, layout, Region(0, 10, 10)));
    scene->setTerrainTile(Region(0, 11, 10), generateTerrainTile(source, layout, Region(0, 11, 10)));
    scene->setTerrainTile(Region(0, 2, 2), generateTerrainTile(source, layout, Region(0, 2, 2)));
    RenderFrame();
    EXPECT_EQ(scene->terrainPassStats().nodesPerLevel[0], 0);
    EXPECT_FALSE(scene->terrainAtlas().tileResident(0, 2, 2));
    RenderFrame();
    EXPECT_TRUE(scene->terrainAtlas().tileResident(0, 2, 2));
    EXPECT_GT(scene->terrainPassStats().nodesPerLevel[0], 0);
}

// Regression (PR review): a normal tap on a tile border fell back to the coarser level when the tile containing the
// sample in the primary lookup was not resident, although its west neighbour holds the same sample: the normal changed
// when the east tile streamed in
TEST_F(VulkanTerrain, NormalAtTileBorderUsesTheNeighbour)
{
    const int bs               = 1;
    const TerrainLayout layout = testLayout(16, bs);
    scene->setTerrainPyramid(0, testPyramid(highFrequencyHook, layout));
    // Every level's tile of chunk (0, 0); level 0's tile (1, 0) is missing
    SetPath(layout, highFrequencyHook, 0, 0);
    renderUntilUploaded(*scene);
    const Vec3 cam(230, 40, 128);
    SetCamera(0, cam, lookRotation(direction(0, -45)), 1.0);
    Capture(2);
    const auto& nodes = scene->terrainPass().selectedNodes();
    auto h            = [](int64_t x, int64_t z) { return double(clampTerrainHeight(highFrequencyHook(1, x, z))); };
    std::vector<TerrainProbe> probes;
    std::vector<Vec3> expected;
    for (size_t n = 0; n < nodes.size(); ++n)
    {
        if (nodes[n].level != 0 || nodes[n].quarter || nodes[n].originX != 224) continue;
        // Odd z: the coarser level has no sample there
        for (int gz = 1; gz < 32; gz += 2)
        {
            const int64_t x = 255, z = nodes[n].originZ + gz;
            probes.push_back(TerrainProbe{int(n), 31, gz, true});
            const Vec3 nrm((h(x - 1, z) - h(x + 1, z)) * kHeightUnit, 2.0 * bs,
                           (h(x, z - 1) - h(x, z + 1)) * kHeightUnit);
            expected.push_back(nrm.normalized());
        }
    }
    ASSERT_GT(probes.size(), 10u);
    const auto out = scene->terrainPass().probeVertices(probes);
    for (size_t i = 0; i < probes.size(); ++i)
    {
        EXPECT_EQ(out[i](3), 0.0f);
        for (int c = 0; c < 3; ++c) EXPECT_NEAR(out[i](c), expected[i](c), 1e-5) << "probe " << i;
    }
}

namespace
{
inline int32_t raised200Hook(uint64_t, int64_t, int64_t)
{
    return 200 * 1024;
}
}  // namespace

// Regression (PR review): a quarter (a child out of its range, drawn at the parent's level) was culled with the child's
// bounds, which miss the parent's heights when the levels' data differ (here level 1 raised by 200 m over flat level-0
// tiles): the raised quarters above a low camera looking up were rejected
TEST_F(VulkanTerrain, QuarterCulledWithItsOwnHeights)
{
    const TerrainLayout layout = testLayout(16, 1);
    scene->setTerrainPyramid(0, testPyramid(flatHook, layout));
    SetPath(layout, flatHook, 0, 0);
    for (int z = 0; z < 2; ++z)
        for (int x = 0; x < 2; ++x) scene->setTerrainTile(Region(0, x, z), testTile(flatHook, layout, Region(0, x, z)));
    scene->setTerrainTile(Region(1, 0, 0), testTile(raised200Hook, layout, Region(1, 0, 0)));
    renderUntilUploaded(*scene);
    SetCamera(0, Vec3(100, 5, 100), lookRotation(direction(45, 30)), 1.0);
    Capture(2);
    int quarters = 0;
    for (const auto& n : scene->terrainPass().selectedNodes()) quarters += n.level == 1 && n.quarter ? 1 : 0;
    EXPECT_GT(quarters, 0);
}

TEST_F(VulkanTerrain, PyramidOnly)
{
    const TerrainLayout layout = testLayout(32, 2);
    scene->setTerrainPyramid(0, testPyramid(slopeRidgeHook, layout));
    renderUntilUploaded(*scene);
    SetCamera(0, Vec3(4000, 600, 4000), lookRotation(direction(30, -35)), 1.0);
    const auto img = Capture(2);
    const auto& s  = scene->terrainPassStats();
    EXPECT_GT(s.nodes, 0);
    for (int k = 0; k < layout.streamedLevels(); ++k) EXPECT_EQ(s.nodesPerLevel[size_t(k)], 0) << k;
    // The ground fills the lower part of the view
    EXPECT_LT(countColor(img, kClear, 0), width * height / 2);
    EXPECT_EQ(scene->statistics().terrainNodes, uint32_t(s.nodes));
}

#ifdef SAIGA_ASSERTS
// Regression (PR review): a probe after clearTerrain wrote descriptors of the cleared set's destroyed buffers
TEST_F(VulkanTerrain, ProbeNeedsTheFramesSet)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const TerrainLayout layout = testLayout(16, 2);
    scene->setTerrainPyramid(0, testPyramid(flatHook, layout));
    renderUntilUploaded(*scene);
    SetCamera(0, Vec3(1000, 100, 1000), lookRotation(direction(0, -45)), 1.0);
    Capture(2);
    ASSERT_GT(scene->terrainPass().selectedNodes().size(), 0u);
    EXPECT_EQ(scene->terrainPass().probeVertices({TerrainProbe{0, 0, 0}}).size(), 1u);
    scene->clearTerrain();
    EXPECT_DEATH(scene->terrainPass().probeVertices({TerrainProbe{0, 0, 0}}), "terrain set changed");
    // A new set without a frame
    scene->setTerrainPyramid(0, testPyramid(flatHook, layout));
    EXPECT_DEATH(scene->terrainPass().probeVertices({TerrainProbe{0, 0, 0}}), "terrain set changed");
    renderUntilUploaded(*scene);
}
#endif

// Regression: a non-finite camera reached integer conversions (UB); RenderScene draws no terrain for it. A huge finite
// height is clamped in double before its conversion (the frame anchor) and draws without a level-0 node
TEST_F(VulkanTerrain, NonFiniteAndHugeCameras)
{
    const TerrainLayout layout = testLayout(32, 2);
    scene->setTerrainPyramid(0, testPyramid(slopeRidgeHook, layout));
    renderUntilUploaded(*scene);
    SetCamera(0, Vec3(4000, 600, 4000), lookRotation(direction(30, -35)), 1.0);
    Capture(2);
    ASSERT_GT(scene->terrainPassStats().nodes, 0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    SetCamera(0, Vec3(nan, 600, 4000), lookRotation(direction(30, -35)), 1.0);
    Capture(2);
    EXPECT_EQ(scene->terrainPassStats().nodes, 0);
    EXPECT_EQ(scene->statistics().terrainNodes, 0u);
    SetCamera(0, Vec3(4000, 1e300, 4000), lookRotation(direction(30, -35)), 1.0);
    Capture(2);
    EXPECT_EQ(scene->terrainPassStats().nodesPerLevel[0], 0);
}

// Remove and set tiles in consecutive frames (a layer reused one frame after its removal): validation clean
TEST_F(VulkanTerrain, TileRemoveAndLayerReuseInFlight)
{
    TerrainLayout layout = testLayout(16, 2);
    layout.tilesPerLevel = 2;
    scene->setTerrainPyramid(0, testPyramid(slopeRidgeHook, layout));
    SetCamera(0, Vec3(300, 120, 300), lookRotation(direction(0, -60)), 1.0);
    for (int i = 0; i < 12; ++i)
    {
        const RegionId a = Region(0, i % 3, 1), b = Region(0, (i + 1) % 3, 1);
        scene->removeTerrainTile(Region(0, (i + 2) % 3, 1));
        scene->setTerrainTile(a, testTile(slopeRidgeHook, layout, a));
        scene->setTerrainTile(b, testTile(slopeRidgeHook, layout, b));
        for (int k = 1; k < layout.streamedLevels(); ++k)
            scene->setTerrainTile(Region(k, 0, 0), testTile(slopeRidgeHook, layout, Region(k, 0, 0)));
        RenderFrame();
    }
    renderUntilUploaded(*scene);
    Capture(2);
    EXPECT_EQ(scene->terrainAtlasStats().tilesSet[0], 2);
}

// A patched tile renders pixel-equal to a full setTerrainTile with the patched data
TEST_F(VulkanTerrain, PatchedImageEqualsFullTile)
{
    const TerrainLayout layout = testLayout(16, 2);
    scene->setTerrainPyramid(0, testPyramid(flatHook, layout));
    for (int k = 1; k < layout.streamedLevels(); ++k)
        scene->setTerrainTile(Region(k, 0, 0), testTile(flatHook, layout, Region(k, 0, 0)));
    const RegionId r = Region(0, 0, 0);
    auto data        = testTile(flatHook, layout, r);
    scene->setTerrainTile(r, data);
    renderUntilUploaded(*scene);
    SetCamera(0, Vec3(140, 60, 60), lookRotation(direction(80, -35)), 1.0);
    Capture(2);
    // A hill of 40 x 30 samples
    const int x0 = 20, z0 = 30, w = 40, h = 30;
    std::vector<int32_t> patch(size_t(w * h));
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) patch[size_t(j * w + i)] = (std::min(i, w - 1 - i) + std::min(j, h - 1 - j)) * 700;
    scene->patchTerrainTile(r, x0, z0, w, h, patch.data());
    renderUntilUploaded(*scene);
    const auto patched = Capture(3);
    auto full          = std::make_shared<std::vector<int32_t>>(*data.heights);
    const int n        = layout.samplesPerSide();
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) (*full)[size_t((z0 + j) * n + x0 + i)] = patch[size_t(j * w + i)];
    TerrainTileData replaced;
    replaced.heights = full;
    terrainHeightRange(*full, replaced.minHeight, replaced.maxHeight);
    scene->setTerrainTile(r, replaced);
    renderUntilUploaded(*scene);
    const auto reference = Capture(3);
    EXPECT_LT(countColor(patched, kClear, 0), width * height);
    int differing = 0;
    for (int y = 0; y < patched.h; ++y)
        for (int x = 0; x < patched.w; ++x)
            for (int c = 0; c < 3; ++c)
                if (patched(y, x)(c) != reference(y, x)(c))
                {
                    differing++;
                    break;
                }
    EXPECT_EQ(differing, 0);
}

// The water plane at the sea level (camera-relative: seaLevel - refHeight, minus cameraRel.y); none without a sea level
TEST_F(VulkanTerrain, WaterAtSeaLevel)
{
    const TerrainLayout layout = testLayout(16, 2);
    // The terrain 10 m below the sea level 2 m
    auto deep = [](uint64_t, int64_t, int64_t) { return int32_t(-10 * 1024); };
    scene->setTerrainPyramid(0, testPyramid(deep, layout, 2048));
    renderUntilUploaded(*scene);
    const Vec3 cam(2000.5, 25, 2000.5);
    SetCamera(0, cam, lookRotation(direction(0, -40)), 1.0);
    const auto img = Capture(2);
    EXPECT_TRUE(scene->terrainPassStats().water);
    const auto& nodes = scene->terrainPass().selectedNodes();
    int water         = -1;
    for (size_t i = 0; i < nodes.size(); ++i)
        if (nodes[i].water) water = int(i);
    ASSERT_GE(water, 0);
    const auto v = scene->terrainPass().probeVertices({TerrainProbe{water, 8, 8}});
    EXPECT_NEAR(double(v[0](1)) + cam.y(), 2.0, 1e-4);
    // The view is filled by water (and terrain)
    EXPECT_EQ(countColor(img, kClear, 0), 0);
    // Regression (CodeRabbit, S20): a huge finite x reached int64_t(floor(x / 4096)) for the water centre (UB); it is
    // clamped like the frame anchor (the plane is then far from the camera and culled)
    for (double x : {1e30, -1e30})
    {
        SetCamera(0, Vec3(x, 25, 2000.5), lookRotation(direction(0, -40)), 1.0);
        Capture(2);
        EXPECT_FALSE(scene->terrainPassStats().water) << x;
    }
}

TEST_F(VulkanTerrain, NoWaterWithoutSeaLevel)
{
    const TerrainLayout layout = testLayout(16, 2);
    auto deep                  = [](uint64_t, int64_t, int64_t) { return int32_t(-10 * 1024); };
    scene->setTerrainPyramid(0, testPyramid(deep, layout, std::nullopt));
    renderUntilUploaded(*scene);
    SetCamera(0, Vec3(2000.5, 25, 2000.5), lookRotation(direction(0, -40)), 1.0);
    Capture(2);
    EXPECT_FALSE(scene->terrainPassStats().water);
    for (const auto& n : scene->terrainPass().selectedNodes()) EXPECT_FALSE(n.water);
    // Water off by the settings
    scene->setTerrainPyramid(0, testPyramid(deep, layout, 2048));
    renderUntilUploaded(*scene);
    PrototypeSettings s = scene->prototypeSettings();
    s.water             = false;
    scene->setPrototypeSettings(s);
    Capture(2);
    EXPECT_TRUE(scene->terrainPassStats().water);  // selected, not drawn
}

// ---------------------------------------------------------------------------------------------------------------------
// Step 5: the grass (GrassPass)

namespace
{
GrassMaskData constantMask(const TerrainLayout& layout, uint8_t value)
{
    GrassMaskData m;
    m.size    = layout.grassMaskSize();
    m.density = std::make_shared<std::vector<uint8_t>>(size_t(m.size) * size_t(m.size), value);
    return m;
}
ChunkId Chunk(int32_t x, int32_t z)
{
    ChunkId c;
    c.x = x;
    c.z = z;
    return c;
}
}  // namespace

// No clumps without a mask or with a zero mask
TEST_F(VulkanTerrain, GrassNeedsMask)
{
    const TerrainLayout layout = testLayout(16, 2);
    scene->setPrototypeEnvironment(PrototypeSettings(), 1);
    const Vec3 cam(300.5, 2, 300.5);
    fillTestTerrain(*scene, layout, TerrainHeightSource::fromHook(flatHook, 1), std::nullopt, cam, false);
    SetCamera(0, cam, lookRotation(direction(30, -30)), 1.0);
    const auto bare = Capture(3);
    EXPECT_EQ(scene->grassPass().stats().blocks, 0);
    // Zero masks: blocks are drawn, every clump is discarded
    for (int32_t z = 0; z < 3; ++z)
        for (int32_t x = 0; x < 3; ++x) scene->setGrassMask(Chunk(x, z), constantMask(layout, 0));
    renderUntilUploaded(*scene);
    const auto zero = Capture(3);
    EXPECT_GT(scene->grassPass().stats().blocks, 0);
    // Regression (CodeRabbit, S20): a NaN grass radius passed the "<= 0" guard and reached int32_t conversions (UB):
    // no grass
    {
        PrototypeSettings nanRadius  = scene->prototypeSettings();
        const PrototypeSettings keep = nanRadius;
        nanRadius.grassRadius        = std::numeric_limits<decltype(nanRadius.grassRadius)>::quiet_NaN();
        scene->setPrototypeSettings(nanRadius);
        Capture(2);
        EXPECT_EQ(scene->grassPass().stats().blocks, 0);
        scene->setPrototypeSettings(keep);
    }
    int differing = 0;
    for (int y = 0; y < bare.h; ++y)
        for (int x = 0; x < bare.w; ++x) differing += bare(y, x) != zero(y, x) ? 1 : 0;
    EXPECT_EQ(differing, 0);
    // Full masks: grass
    for (int32_t z = 0; z < 3; ++z)
        for (int32_t x = 0; x < 3; ++x) scene->setGrassMask(Chunk(x, z), constantMask(layout, 255));
    renderUntilUploaded(*scene);
    const auto full = Capture(3);
    differing       = 0;
    for (int y = 0; y < bare.h; ++y)
        for (int x = 0; x < bare.w; ++x) differing += bare(y, x) != full(y, x) ? 1 : 0;
    EXPECT_GT(differing, width * height / 20);
    // A removed mask: no blocks of that chunk
    for (int32_t z = 0; z < 3; ++z)
        for (int32_t x = 0; x < 3; ++x) scene->removeGrassMask(Chunk(x, z));
    Capture(2);
    EXPECT_EQ(scene->grassPass().stats().blocks, 0);
}

// The grass is chunk-relative: the image is invariant under a 256 m camera and content offset
TEST_F(VulkanTerrain, GrassChunkRelative)
{
    const TerrainLayout layout = testLayout(16, 1);
    scene->setPrototypeEnvironment(PrototypeSettings(), 1);
    auto periodic = [](uint64_t, int64_t x, int64_t z)
    { return int32_t(triangleWave(x, 256) * 12 + triangleWave(z + 2 * x, 128) * 8 + 2048); };
    const auto source = TerrainHeightSource::fromHook(periodic, 1);
    std::vector<TemplatedImage<ucvec4>> images;
    for (int k = 0; k < 2; ++k)
    {
        const Vec3 eye(1700.3 + 256 * k, 0, 1500.6);
        const Vec3 cam = eye + Vec3(0, surfaceAt(periodic, 1, 1, eye.x(), eye.z()) + 2, 0);
        fillTestTerrain(*scene, layout, source, std::nullopt, cam, true);
        SetCamera(0, cam, lookRotation(direction(20, -45)), 40.0 * kPi / 180);
        images.push_back(Capture(3));
        EXPECT_GT(scene->grassPass().stats().blocks, 0);
    }
    int differing = 0;
    for (int y = 0; y < images[0].h; ++y)
        for (int x = 0; x < images[0].w; ++x) differing += images[0](y, x) != images[1](y, x) ? 1 : 0;
    EXPECT_EQ(differing, 0);
    EXPECT_LT(countColor(images[0], kClear, 0), width * height / 10);
}
