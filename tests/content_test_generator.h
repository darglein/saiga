/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine test generator of the content hashes (docs/engine_design.md §12.1; S23a, Scope 11): TestTerrainGenerator
// "test_terrain", versions 1 and 2, used by both open-world templates of the test setup (plan() branches on the
// template name). Integer only: content code (the ContentLint test scans this header); header only.
//
// - height: fbm plus ridged noise and a value-noise detail, river channels carved below the sea level from the rivers
//   plan grid (filled first in plan(), so height() can be called while placing sites);
// - slots test_tree (TombstoneOnEdit, meshes test/tree_a and test/tree_b; a jittered grid of 2 m cells, 128 x 128 =
//   16384 candidates per chunk, about one in 16 accepted, none below the sea) and test_rock (ReSnapOnEdit, test/rock;
//   an 8 m grid, 1024 candidates);
// - grass from slope and height;
// - the plan of "overworld": site 0 test_ruin at (1100, 1300) with one quarter turn (chunk (4, 5); it creates one
//   test_dungeon instance), site 1 test_shrine at (1536, 1536) with two quarter turns (across the corner of chunks
//   (5..6, 5..6)), further test_shrine sites on a 4 km grid (at most 64) with random quarter turns where they fit;
//   every baseHeight is the natural height at the site origin; it lists one instance of test_valley;
// - the plan of "test_valley" (base spacing 1, 16 chunks): two test_shrine sites, one with three quarter turns;
// - version 2 changes only the placement (another jitter seed) and the grass threshold; plan and height equal v1.
// The variants (tests only, never in the committed file): PlanChangeV2 moves site 1 by 256 m in v2 (the same space
// table, other heights), TableChangeV2 lists a second test_valley instance in v2, ReleasedChangeV1 moves site 1 in v1.

#pragma once

#include "saiga/engine/content/Generator.h"
#include "saiga/engine/content/gen/FixedMath.h"
#include "saiga/engine/content/gen/IntNoise.h"
#include "saiga/engine/content/gen/SitePatch.h"
#include "saiga/engine/content/gen/TrigTable.h"
#include "saiga/engine/core/Hash.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ContentTest
{
using namespace Saiga;
using namespace Saiga::Engine;

enum class Variant : uint8_t
{
    Committed,
    PlanChangeV2,
    TableChangeV2,
    ReleasedChangeV1
};

class TestTerrainGenerator final : public Generator
{
   public:
    explicit TestTerrainGenerator(uint16_t maxVersion, Variant variant = Variant::Committed)
        : maxVersion(maxVersion), variant(variant)
    {
    }

    const char* name() const override { return "test_terrain"; }
    uint16_t version() const override { return maxVersion; }
    std::vector<GeneratorSlot> slots() const override
    {
        return {{"test_tree", TombstoneOnEdit, {"test/tree_a", "test/tree_b"}},
                {"test_rock", ReSnapOnEdit, {"test/rock"}}};
    }

    void plan(uint16_t version, WorldPlan& plan) const override
    {
        const int32_t size = plan.space.sizeChunks;
        // The grids height() reads, first
        plan.rivers.cellSize = 512;
        plan.rivers.width    = size / 2;
        plan.rivers.height   = size / 2;
        plan.rivers.values.assign(uint32_t(plan.rivers.width) * uint32_t(plan.rivers.height), 0);
        for (int32_t z = 0; z < plan.rivers.height; ++z)
            for (int32_t x = 0; x < plan.rivers.width; ++x)
                plan.rivers.values[uint32_t(z * plan.rivers.width + x)] =
                    contentHash64(plan.seed, uint64_t(x), uint64_t(z), 9, 0) % 23 == 0 ? 1 : 0;
        plan.biomes.cellSize = 2048;
        plan.biomes.width    = (size + 7) / 8;
        plan.biomes.height   = (size + 7) / 8;
        plan.biomes.values.assign(uint32_t(plan.biomes.width) * uint32_t(plan.biomes.height), 0);
        for (int32_t z = 0; z < plan.biomes.height; ++z)
            for (int32_t x = 0; x < plan.biomes.width; ++x)
                plan.biomes.values[uint32_t(z * plan.biomes.width + x)] =
                    int32_t(contentHash64(plan.seed, uint64_t(x), uint64_t(z), 10, 0) % 4);

        const bool moveSite1 = (variant == Variant::PlanChangeV2 && version >= 2) ||
                               (variant == Variant::ReleasedChangeV1 && version == 1);
        if (plan.space.name == "overworld")
        {
            addSite(version, plan, 0, "test_ruin", 1100, 1300, 1, 16);
            addSite(version, plan, 1, "test_shrine", moveSite1 ? 1536 + 256 : 1536, 1536, 2, 8);
            // test_shrine on a 4 km grid where it fits
            const int64_t cells = int64_t(size) * 256 / 4096;
            for (int64_t gz = 0; gz < cells; ++gz)
                for (int64_t gx = 0; gx < cells; ++gx)
                {
                    const uint64_t h = contentHash64(plan.seed, uint64_t(gx), uint64_t(gz), 11, 0);
                    const int64_t x  = gx * 4096 + 64 + int64_t(h % 3968);
                    const int64_t z  = gz * 4096 + 64 + int64_t((h >> 16) % 3968);
                    const uint8_t q  = uint8_t((h >> 32) & 3);
                    if (fits(plan, x, z, 8))
                        addSite(version, plan, uint32_t(2 + gz * cells + gx), "test_shrine", x, z, q, 8);
                }
            SpaceInstance valley;
            valley.templateIndex = 1;
            valley.seed          = hashCombine(hashCombine(plan.seed, fnv1a64("listed")), 0);
            plan.spaces.push_back(valley);
            if (variant == Variant::TableChangeV2 && version >= 2)
            {
                valley.seed = hashCombine(hashCombine(plan.seed, fnv1a64("listed")), 1);
                plan.spaces.push_back(valley);
            }
        }
        else if (plan.space.name == "test_valley")
        {
            addSite(version, plan, 0, "test_shrine", 1000, 1200, 3, 8);
            addSite(version, plan, 1, "test_shrine", 3000, 2500, 0, 8);
        }
    }

    int32_t height(uint16_t, const WorldPlan& plan, int64_t x, int64_t z) const override
    {
        const uint64_t s = plan.seed;
        int64_t h        = floorDiv(int64_t(fbm(s, x, z, 11, 6)) * 60 * 1024, 65536);
        h += floorDiv(int64_t(ridged(contentHash64(s, 1, 0, 0, 0), x, z, 10, 4, 2)) * 25 * 1024, 65536);
        h += mulShift(valueNoise(contentHash64(s, 2, 0, 0, 0), x, z, 4), 2 * 1024, 16);
        h += 12 * 1024;
        // A river cell of 512 m carves a channel toward its centre: the depth fades with the distance (smooth)
        if (plan.rivers.at(floorDiv(x, 512), floorDiv(z, 512)) != 0)
        {
            const int64_t dx = floorMod(x, 512) - 256, dz = floorMod(z, 512) - 256;
            const int64_t d = int64_t(isqrt64(uint64_t(dx * dx + dz * dz)));
            if (d < 256)
            {
                const int64_t w      = smoothQ16((256 - d) * 256);
                const int64_t sea    = plan.space.seaLevel ? *plan.space.seaLevel : 0;
                const int64_t carved = sea - 3 * 1024;
                h                    = lerpQ16(h, h < carved ? h : carved, w);
            }
        }
        return int32_t(clampI<int64_t>(h, -kTerrainHeightLimit, kTerrainHeightLimit));
    }

    bool candidate(uint16_t version, const WorldPlan& plan, ChunkId chunk, uint32_t slot, uint32_t counter,
                   Candidate& out) const override
    {
        const uint64_t jitter = contentHash64(plan.seed, version >= 2 ? 2 : 1, 0x7E57, 0, 0);
        const uint64_t h = contentHash64(jitter, slot, uint64_t(int64_t(chunk.x)), uint64_t(int64_t(chunk.z)), counter);
        const int64_t sea = plan.space.seaLevel ? *plan.space.seaLevel : 0;
        if (slot == 0)
        {
            // test_tree: 2 m cells, about one in 16
            if ((h & 15) != 0) return false;
            const int64_t i = counter & 127, j = counter >> 7;
            out.x        = int32_t(i * 512 + int64_t((h >> 8) % 512));
            out.z        = int32_t(j * 512 + int64_t((h >> 20) % 512));
            out.meshSlot = uint16_t((h >> 40) & 1);
            out.yaw      = uint16_t(h >> 32);
            out.scale    = uint16_t(900 + (h >> 48) % 256);
            out.flags    = 0;
        }
        else
        {
            // test_rock: 8 m cells, about one in 4, offset by up to 3 m in a direction from the trig table
            if ((h & 3) != 0) return false;
            const int64_t i = counter & 31, j = counter >> 5;
            const uint32_t angle = uint32_t((h >> 8) & 0x1FFFF);
            const int64_t r      = int64_t((h >> 32) % 768);
            out.x                = int32_t(i * 2048 + 1024 + mulShift(cosQ30(angle), r, 30));
            out.z                = int32_t(j * 2048 + 1024 + mulShift(sinQ30(angle), r, 30));
            out.meshSlot         = 0;
            out.yaw              = uint16_t(h >> 16);
            out.scale            = uint16_t(1024 + (h >> 40) % 512);
            out.flags            = uint32_t(1 + ((h >> 50) & 1) * 2);
        }
        // Nothing under the sea (the natural height of the sample below the candidate)
        const int64_t mx = int64_t(chunk.x) * 256 + out.x / 256, mz = int64_t(chunk.z) * 256 + out.z / 256;
        return height(version, plan, mx, mz) > sea + 512;
    }

    uint32_t candidatesPerChunk(uint32_t slot) const override { return slot == 0 ? 16384 : 1024; }
    uint32_t keptCount(uint32_t slot, int level) const override
    {
        return level >= 15 ? 0 : candidatesPerChunk(slot) >> level;
    }

    void grassMask(uint16_t version, const WorldPlan& plan, ChunkId chunk, GrassMask& out) const override
    {
        const int32_t spacing = plan.space.baseSpacing == 1 ? 1 : 2;
        const int32_t n       = 256 / spacing;
        out.size              = n;
        out.density.assign(uint32_t(n) * uint32_t(n), 0);
        // The natural heights at the cell corners
        std::vector<int32_t> h(uint32_t(n + 1) * uint32_t(n + 1));
        const int64_t x0 = int64_t(chunk.x) * 256, z0 = int64_t(chunk.z) * 256;
        for (int32_t j = 0; j <= n; ++j)
            for (int32_t i = 0; i <= n; ++i)
                h[uint32_t(j * (n + 1) + i)] =
                    height(version, plan, x0 + int64_t(i) * spacing, z0 + int64_t(j) * spacing);
        const int64_t sea       = plan.space.seaLevel ? *plan.space.seaLevel : 0;
        const int64_t threshold = int64_t(version >= 2 ? 600 : 700) * spacing;
        const uint64_t s        = contentHash64(plan.seed, 5, 0, 0, 0);
        for (int32_t j = 0; j < n; ++j)
            for (int32_t i = 0; i < n; ++i)
            {
                const int64_t h00 = h[uint32_t(j * (n + 1) + i)];
                const int64_t dx  = h[uint32_t(j * (n + 1) + i + 1)] - h00;
                const int64_t dz  = h[uint32_t((j + 1) * (n + 1) + i)] - h00;
                if ((dx < 0 ? -dx : dx) + (dz < 0 ? -dz : dz) > threshold || h00 < sea + 400) continue;
                const int64_t v = valueNoise(s, x0 + int64_t(i) * spacing, z0 + int64_t(j) * spacing, 4);
                out.density[uint32_t(j * n + i)] = uint8_t(128 + floorDiv(v * 127, 65536));
            }
    }

   private:
    uint16_t maxVersion = 2;
    Variant variant     = Variant::Committed;

    // A square footprint of half size r rotated anywhere stays the same square: inside the space with a margin and
    // apart from every site (closed bounds, as the pipeline tests them)
    static bool fits(const WorldPlan& plan, int64_t x, int64_t z, int64_t r)
    {
        const int64_t size = int64_t(plan.space.sizeChunks) * 256;
        if (x - r < 0 || z - r < 0 || x + r > size || z + r > size) return false;
        for (const Site& s : plan.sites)
        {
            int64_t b[4];
            rotateBounds(s.footprint, s.quarterTurns, b);
            if (x - r <= s.x + b[2] && s.x + b[0] <= x + r && z - r <= s.z + b[3] && s.z + b[1] <= z + r) return false;
        }
        return true;
    }

    void addSite(uint16_t version, WorldPlan& plan, uint32_t key, const char* prefab, int64_t x, int64_t z, uint8_t q,
                 int32_t half) const
    {
        Site s;
        s.key           = key;
        s.prefab        = prefab;
        s.prefabVersion = 1;
        s.x             = x;
        s.z             = z;
        s.quarterTurns  = q;
        s.baseHeight    = height(version, plan, x, z);
        s.footprint[0]  = -half;
        s.footprint[1]  = -half;
        s.footprint[2]  = half;
        s.footprint[3]  = half;
        plan.sites.push_back(s);
    }
};

}  // namespace ContentTest
