/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// RenderScene's terrain (docs/engine_design.md §10.1, §12.2; S20) driven with test tiles of the pure generation and the
// integer hooks of terrain_test_hooks.h: the GPU terrain set (TerrainAtlas: uploads, the byte cap, patches, rebuilds,
// limits, memory, node bounds).

#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/TerrainAtlas.h"
#include "saiga/vulkan/scene/TerrainTiles.h"

#include "gtest/gtest.h"

#include <cmath>
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
