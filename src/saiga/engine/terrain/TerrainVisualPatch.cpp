/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/TerrainVisualPatch.h"

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/terrain/TerrainEditStore.h"

#include <algorithm>
#include <memory>

namespace Saiga
{
namespace Engine
{
namespace
{
int32_t clampSum(int32_t base, int32_t delta)
{
    const int64_t v = int64_t(base) + delta;
    return int32_t(std::min<int64_t>(std::max<int64_t>(v, -kTerrainHeightLimit), kTerrainHeightLimit));
}

// Writes clamp(base + delta) for every sample of the grid (i0 + a * step, k0 + b * step) of the chunk range that lies
// in an allocated block (out holds a copy of base; idempotent: a sample on a chunk border is visited once per chunk
// that contains it, each visit computes the value from the base)
void applyRecords(const TerrainEditStore& edits, SpaceId space, int64_t x0, int64_t z0, int64_t x1, int64_t z1,
                  int64_t i0, int64_t k0, int64_t step, int w, int h, const int32_t* base, int32_t* out)
{
    const int64_t S = edits.chunkSamples(space);
    for (const auto& [c, e] : edits.chunks(space))
    {
        if (e.nonZeroSamples == 0 || c.x < x0 || c.x > x1 || c.z < z0 || c.z > z1) continue;
        // The chunk's closed range on the tile's grid
        const int64_t ci0 = int64_t(c.x) * S, ck0 = int64_t(c.z) * S;
        const int64_t a0 = std::max<int64_t>(0, (ci0 - i0 + step - 1) / step);
        const int64_t a1 = std::min<int64_t>(w - 1, (ci0 + S - i0) / step);
        const int64_t b0 = std::max<int64_t>(0, (ck0 - k0 + step - 1) / step);
        const int64_t b1 = std::min<int64_t>(h - 1, (ck0 + S - k0) / step);
        for (int64_t b = b0; b <= b1; ++b)
        {
            const int64_t k               = k0 + b * step;
            int64_t lastKey               = -1;
            const TerrainEditBlock* block = nullptr;
            for (int64_t a = a0; a <= a1; ++a)
            {
                const int64_t i   = i0 + a * step;
                const int64_t key = int64_t(TerrainEditStore::blockKey(i, k));
                if (key != lastKey)
                {
                    block   = edits.block(space, i, k);
                    lastKey = key;
                }
                if (!block) continue;
                const int32_t d  = block->delta[(k & 15) * 16 + (i & 15)];
                const size_t idx = size_t(b * w + a);
                out[idx]         = clampSum(base[idx], d);
            }
        }
    }
}
}  // namespace

bool TerrainVisualPatch::hasRecords(const TerrainEditStore& edits, SpaceId space, int64_t x0, int64_t z0, int64_t x1,
                                    int64_t z1)
{
    if (space >= edits.spaceCount() || edits.nonZeroSamples(space) == 0) return false;
    for (const auto& [c, e] : edits.chunks(space))
        if (e.nonZeroSamples > 0 && c.x >= x0 && c.x <= x1 && c.z >= z0 && c.z <= z1) return true;
    return false;
}

bool TerrainVisualPatch::mergeTile(const TerrainEditStore& edits, const RegionId& region,
                                   const Vulkan::TerrainLayout& layout, const Vulkan::TerrainTileData& base,
                                   Vulkan::TerrainTileData& out)
{
    const int64_t span = int64_t(1) << region.level;
    const int64_t x0 = int64_t(region.x) * span, z0 = int64_t(region.z) * span;
    if (!hasRecords(edits, region.space, x0, z0, x0 + span - 1, z0 + span - 1)) return false;
    const int N     = layout.samplesPerSide();
    const int64_t S = edits.chunkSamples(region.space);
    auto heights    = std::make_shared<std::vector<int32_t>>(*base.heights);
    applyRecords(edits, region.space, x0, z0, x0 + span - 1, z0 + span - 1, x0 * S, z0 * S, span, N, N,
                 base.heights->data(), heights->data());
    out.minHeight = *std::min_element(heights->begin(), heights->end());
    out.maxHeight = *std::max_element(heights->begin(), heights->end());
    out.heights   = heights;
    return true;
}

bool TerrainVisualPatch::mergePyramid(const TerrainEditStore& edits, SpaceId space,
                                      const Vulkan::TerrainPyramidData& base, Vulkan::TerrainPyramidData& out)
{
    const int64_t size = edits.sizeChunks(space);
    if (!hasRecords(edits, space, 0, 0, size - 1, size - 1)) return false;
    const int P        = base.layout.pyramidSamplesPerSide();
    const int64_t step = Vulkan::TerrainLayout::pyramidSpacing / edits.spacing(space);
    auto heights       = std::make_shared<std::vector<int32_t>>(*base.heights);
    applyRecords(edits, space, 0, 0, size - 1, size - 1, 0, 0, step, P, P, base.heights->data(), heights->data());
    out           = base;
    out.minHeight = *std::min_element(heights->begin(), heights->end());
    out.maxHeight = *std::max_element(heights->begin(), heights->end());
    out.heights   = heights;
    return true;
}

void TerrainVisualPatch::mergedHeights(const TerrainEditStore& edits, SpaceId space, int64_t i0, int64_t k0,
                                       int64_t step, int w, int h, const int32_t* base, size_t baseStride, int32_t* out)
{
    for (int b = 0; b < h; ++b)
    {
        const int64_t k               = k0 + b * step;
        int64_t lastKey               = -1;
        const TerrainEditBlock* block = nullptr;
        for (int a = 0; a < w; ++a)
        {
            const int64_t i   = i0 + a * step;
            const int64_t key = int64_t(TerrainEditStore::blockKey(i, k));
            if (key != lastKey)
            {
                block   = edits.block(space, i, k);
                lastKey = key;
            }
            const int32_t v        = base[size_t(b) * baseStride + size_t(a)];
            out[size_t(b * w + a)] = block ? clampSum(v, block->delta[(k & 15) * 16 + (i & 15)]) : v;
        }
    }
}

}  // namespace Engine
}  // namespace Saiga
