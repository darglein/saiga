/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainGeneration (docs/engine_design.md §12.2; S20): the engine's terrain generation, thin wrappers over the pure
// functions of vulkan/scene/TerrainTiles (D2). TerrainSystem builds each space's height source once with heightSource;
// S23a replaces only heightSource by its pipeline source.

#pragma once

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/world/SpaceTemplate.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

namespace Saiga
{
namespace Engine
{
// The space's height source: the prototype hook with the space seed (clamped, fromHook); empty without terrain
SAIGA_ENGINE_API TerrainHeightSource heightSource(const SpaceTemplate& space, uint64_t seed);
SAIGA_ENGINE_API Vulkan::TerrainTileData generateTile(const TerrainHeightSource& source,
                                                      const Vulkan::TerrainLayout& layout, const RegionId& region);
SAIGA_ENGINE_API Vulkan::TerrainPyramidData generatePyramid(const TerrainHeightSource& source,
                                                            const Vulkan::TerrainLayout& layout,
                                                            std::optional<int32_t> seaLevel);
// The transitional grass rule over a level-0 tile (S23b: generator masks)
SAIGA_ENGINE_API Vulkan::GrassMaskData generateGrassMask(const Vulkan::TerrainTileData& level0,
                                                         const Vulkan::TerrainLayout& layout,
                                                         std::optional<int32_t> seaLevel, double minNormalY,
                                                         double minAboveSea);

}  // namespace Engine
}  // namespace Saiga
