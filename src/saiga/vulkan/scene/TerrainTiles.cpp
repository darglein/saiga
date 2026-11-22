/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/TerrainTiles.h"

#include "saiga/core/util/assert.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Vulkan
{
void terrainHeightRange(const std::vector<int32_t>& heights, int32_t& minHeight, int32_t& maxHeight)
{
    SAIGA_ASSERT(!heights.empty());
    auto [lo, hi] = std::minmax_element(heights.begin(), heights.end());
    minHeight     = *lo;
    maxHeight     = *hi;
    SAIGA_ASSERT(minHeight >= -kTerrainHeightLimit && maxHeight <= kTerrainHeightLimit,
                 "terrain generation: a height source returned a value beyond kTerrainHeightLimit");
}

TerrainTileData generateTerrainTile(const TerrainHeightSource& source, const TerrainLayout& layout,
                                    const RegionId& region)
{
    SAIGA_ASSERT(source, "generateTerrainTile: no height source");
    SAIGA_ASSERT(region.level < layout.streamedLevels(), "generateTerrainTile: not a streamed level");
    const int n           = layout.samplesPerSide();
    const int64_t spacing = layout.levelSpacing(region.level);
    const int64_t size    = int64_t(256) << region.level;
    const int64_t x0 = int64_t(region.x) * size, z0 = int64_t(region.z) * size;
    auto heights = std::make_shared<std::vector<int32_t>>(size_t(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) (*heights)[size_t(j) * n + i] = source(x0 + i * spacing, z0 + j * spacing);
    TerrainTileData t;
    terrainHeightRange(*heights, t.minHeight, t.maxHeight);
    t.heights = std::move(heights);
    return t;
}

TerrainPyramidData generateTerrainPyramid(const TerrainHeightSource& source, const TerrainLayout& layout,
                                          std::optional<int32_t> seaLevel)
{
    SAIGA_ASSERT(source, "generateTerrainPyramid: no height source");
    const int n     = layout.pyramidSamplesPerSide();
    const int64_t s = TerrainLayout::pyramidSpacing;
    auto make       = [&]()
    {
        std::vector<int32_t> h(size_t(n) * n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) h[size_t(j) * n + i] = source(i * s, j * s);
        return h;
    };
    std::shared_ptr<const std::vector<int32_t>> heights =
        source.cachedGrid ? source.cachedGrid(n, s, make) : std::make_shared<const std::vector<int32_t>>(make());
    SAIGA_ASSERT(heights && heights->size() == size_t(n) * n, "generateTerrainPyramid: a cached grid of another size");
    TerrainPyramidData p;
    p.layout   = layout;
    p.seaLevel = seaLevel;
    terrainHeightRange(*heights, p.minHeight, p.maxHeight);
    p.heights = std::move(heights);
    return p;
}

GrassMaskData prototypeGrassMask(const TerrainTileData& level0, const TerrainLayout& layout,
                                 std::optional<int32_t> seaLevel, double minNormalY, double minAboveSea)
{
    const int n = layout.samplesPerSide();
    const int m = layout.grassMaskSize();
    SAIGA_ASSERT(level0.heights && level0.heights->size() == size_t(n) * n, "prototypeGrassMask: not a level-0 tile");
    const std::vector<int32_t>& h = *level0.heights;
    // The slopes of a triangle in height units per cell -> metres per metre
    const double scale = kHeightUnit / double(layout.baseSpacing);
    auto normalY       = [scale](int64_t dA, int64_t dB)
    {
        const double gx = double(dA) * scale, gz = double(dB) * scale;
        return 1.0 / std::sqrt(1.0 + gx * gx + gz * gz);
    };
    auto density = std::make_shared<std::vector<uint8_t>>(size_t(m) * m, uint8_t(0));
    for (int j = 0; j < m; ++j)
    {
        for (int i = 0; i < m; ++i)
        {
            const int64_t h00 = h[size_t(j) * n + i], h10 = h[size_t(j) * n + i + 1];
            const int64_t h01 = h[size_t(j + 1) * n + i], h11 = h[size_t(j + 1) * n + i + 1];
            // A = (00, 10, 11): slopes h10 - h00, h11 - h10; B = (00, 11, 01): h11 - h01, h01 - h00
            const bool flat =
                normalY(h10 - h00, h11 - h10) >= minNormalY && normalY(h11 - h01, h01 - h00) >= minNormalY;
            const int64_t lowest = std::min({h00, h10, h01, h11});
            const bool dry = !seaLevel || double(lowest) * kHeightUnit >= double(*seaLevel) * kHeightUnit + minAboveSea;
            (*density)[size_t(j) * m + i] = flat && dry ? 255 : 0;
        }
    }
    GrassMaskData g;
    g.size    = m;
    g.density = std::move(density);
    return g;
}

}  // namespace Vulkan
}  // namespace Saiga
