/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The terrain surface rule (docs/engine_design.md §12.2, S20): integer heights in height units (kHeightUnit), the
// height source of a space and the triangulation of a height grid. Shared by the engine (World::terrainHeight), the
// pure tile generation (vulkan/scene/TerrainTiles), the prototype placement (vulkan/world/) and the render tests' CPU
// reference, so every consumer computes the same surface.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/util/assert.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>

namespace Saiga
{
// A height hook: pure, thread safe (called on the streaming pool); x, z in integer metres; the result in height units
using TerrainHeightHook = int32_t (*)(uint64_t seed, int64_t x, int64_t z);

// To [-kTerrainHeightLimit, kTerrainHeightLimit]
inline int32_t clampTerrainHeight(int32_t h)
{
    return std::min(std::max(h, -kTerrainHeightLimit), kTerrainHeightLimit);
}

// What the generation functions and TerrainSystem::sample read (E7): a callable over integer metres, the seed already
// bound. Copyable (pool jobs capture it by value; heavy state behind shared_ptr), thread safe, pure. Its results lie in
// +-kTerrainHeightLimit: fromHook clamps (every hook result passes the clamp), other sources (S23a's integer
// generators) guarantee it; the generation functions assert it (SAIGA_ASSERT).
struct TerrainHeightSource
{
    std::function<int32_t(int64_t x, int64_t z)> height;  // empty: no terrain
    // Optional: a cache of point grids of this source (S20's pyramid: n x n samples at (i, j) * spacing). It returns
    // the cached grid of (n, spacing) or the one make computes (the values are make's either way). The generation
    // pipeline's sources share them through the ContentCache (one pyramid per plan, not per World)
    using Grid = std::shared_ptr<const std::vector<int32_t>>;
    std::function<Grid(int n, int64_t spacing, const std::function<std::vector<int32_t>()>& make)> cachedGrid;
    int32_t operator()(int64_t x, int64_t z) const { return height(x, z); }
    explicit operator bool() const { return bool(height); }
    // clampTerrainHeight(hook(seed, x, z)); a null hook gives an empty source
    static TerrainHeightSource fromHook(TerrainHeightHook hook, uint64_t seed)
    {
        TerrainHeightSource s;
        if (hook) s.height = [hook, seed](int64_t x, int64_t z) { return clampTerrainHeight(hook(seed, x, z)); };
        return s;
    }
};

namespace TerrainSurfaceDetail
{
// The two triangles of a cell: the slopes along fx and fz of the triangle that contains (fx, fz)
inline void slopes(int32_t h00, int32_t h10, int32_t h01, int32_t h11, bool triangleA, int64_t& dA, int64_t& dB)
{
    if (triangleA)
    {
        // (00, 10, 11): fx >= fz
        dA = int64_t(h10) - h00;
        dB = int64_t(h11) - h10;
    }
    else
    {
        // (00, 11, 01): fz > fx
        dA = int64_t(h11) - h01;
        dB = int64_t(h01) - h00;
    }
}
}  // namespace TerrainSurfaceDetail

// The numerator h00 * 2^f + dA * fx + dB * fz of the surface rule in units of 2^-fracBits height units (exact; below
// 2^46 in magnitude for heights within the limit and fracBits <= 20)
inline int64_t terrainSurfaceNumerator(int32_t h00, int32_t h10, int32_t h01, int32_t h11, int64_t fx, int64_t fz,
                                       int fracBits)
{
    SAIGA_ASSERT(fracBits >= 0 && fracBits <= 20, "terrainSurface: fracBits must be in 0..20");
    SAIGA_ASSERT(fx >= 0 && fz >= 0 && fx <= (int64_t(1) << fracBits) && fz <= (int64_t(1) << fracBits),
                 "terrainSurface: the fraction must lie in the cell");
    SAIGA_ASSERT(std::max({std::abs(h00), std::abs(h10), std::abs(h01), std::abs(h11)}) <= kTerrainHeightLimit,
                 "terrainSurface: heights beyond kTerrainHeightLimit");
    int64_t dA = 0, dB = 0;
    TerrainSurfaceDetail::slopes(h00, h10, h01, h11, fx >= fz, dA, dB);
    return int64_t(h00) * (int64_t(1) << fracBits) + dA * fx + dB * fz;
}

// The cell (i, j)..(i+1, j+1) of a height grid is split along its diagonal (i, j)-(i+1, j+1) (§12.2, fixed forever):
// triangle A = (00, 10, 11) where fx >= fz, triangle B = (00, 11, 01) where fz > fx; fx, fz in [0, 1] of the cell.
// Integer form: fx, fz in [0, 2^fracBits] (units of 2^-fracBits of a cell), 0 <= fracBits <= 20, heights within
// +-kTerrainHeightLimit (asserted). The int64 numerator h00 * 2^f + dA * fx + dB * fz is below 2^46; the result in
// height units is rounded toward -infinity (floor division, not '>>' on a negative value: implementation defined in
// C++17).
inline int32_t terrainSurfaceUnits(int32_t h00, int32_t h10, int32_t h01, int32_t h11, int64_t fx, int64_t fz,
                                   int fracBits)
{
    const int64_t num = terrainSurfaceNumerator(h00, h10, h01, h11, fx, fz, fracBits);
    const int64_t d   = int64_t(1) << fracBits;
    int64_t q         = num / d;
    if (num % d != 0 && num < 0) q--;
    return int32_t(q);
}

// Double form: the same triangles; for dyadic fx, fz (fracBits <= 20) every product and sum is below 2^53, so the
// result is exactly numerator / 2^fracBits (in height units, not rounded)
inline double terrainSurfaceDouble(int32_t h00, int32_t h10, int32_t h01, int32_t h11, double fx, double fz)
{
    int64_t dA = 0, dB = 0;
    TerrainSurfaceDetail::slopes(h00, h10, h01, h11, fx >= fz, dA, dB);
    return double(h00) + double(dA) * fx + double(dB) * fz;
}

// Triangle indices of an n x n grid of cells ((n + 1)^2 vertices, vertex (i, j) at j * (n + 1) + i, j along z) with
// this diagonal, counter-clockwise seen from above (the cross product of the edges points to +y; the render grids use
// it). n <= 255 (16-bit indices)
inline void terrainGridTriangles(int n, std::vector<uint16_t>& indices)
{
    SAIGA_ASSERT(n > 0 && n <= 255, "terrainGridTriangles: 1 <= n <= 255");
    indices.clear();
    indices.reserve(size_t(n) * n * 6);
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            const uint16_t v00 = uint16_t(j * (n + 1) + i);
            const uint16_t v10 = uint16_t(v00 + 1);
            const uint16_t v01 = uint16_t(v00 + n + 1);
            const uint16_t v11 = uint16_t(v01 + 1);
            // A = (00, 10, 11) and B = (00, 11, 01), each wound counter-clockwise from above
            indices.insert(indices.end(), {v00, v11, v10, v00, v01, v11});
        }
    }
}

}  // namespace Saiga
