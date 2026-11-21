/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/TerrainGeneration.h"

#include "saiga/vulkan/scene/TerrainTiles.h"

namespace Saiga
{
namespace Engine
{
TerrainHeightSource heightSource(const SpaceTemplate& space, uint64_t seed)
{
    if (!space.hasTerrain()) return TerrainHeightSource();
    return TerrainHeightSource::fromHook(space.prototypeHeight, seed);
}

Vulkan::TerrainTileData generateTile(const TerrainHeightSource& source, const Vulkan::TerrainLayout& layout,
                                     const RegionId& region)
{
    return Vulkan::generateTerrainTile(source, layout, region);
}

Vulkan::TerrainPyramidData generatePyramid(const TerrainHeightSource& source, const Vulkan::TerrainLayout& layout,
                                           std::optional<int32_t> seaLevel)
{
    return Vulkan::generateTerrainPyramid(source, layout, seaLevel);
}

Vulkan::GrassMaskData generateGrassMask(const Vulkan::TerrainTileData& level0, const Vulkan::TerrainLayout& layout,
                                        std::optional<int32_t> seaLevel, double minNormalY, double minAboveSea)
{
    return Vulkan::prototypeGrassMask(level0, layout, seaLevel, minNormalY, minAboveSea);
}

}  // namespace Engine
}  // namespace Saiga
