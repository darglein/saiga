/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/TerrainSurface.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <optional>

namespace Saiga
{
namespace Vulkan
{
/**
 * The pure terrain generation of S20 (docs/engine_design.md §12.2): functions of their arguments only, thread safe, no
 * engine types (D2). The engine's TerrainGeneration wraps them; the renderer tests call them directly.
 */

// The tile of the region (level L < layout.streamedLevels()): the source at regionOrigin + (i, j) * baseSpacing * 2^L
// (integer metres). Coarser levels are point subsamples of the same function: a level-L+1 sample equals the level-L
// sample at the same point (CDLOD morphing relies on it)
SAIGA_VULKAN_API TerrainTileData generateTerrainTile(const TerrainHeightSource& source, const TerrainLayout& layout,
                                                     const RegionId& region);

// The space at 32 m steps (point samples)
SAIGA_VULKAN_API TerrainPyramidData generateTerrainPyramid(const TerrainHeightSource& source,
                                                           const TerrainLayout& layout,
                                                           std::optional<int32_t> seaLevel);

// The transitional grass rule (S23b takes generator masks): 255 where both triangles of the base cell have
// normal.y >= minNormalY and the cell's lowest sample is >= seaLevel + minAboveSea (metres; no sea: no such limit),
// else 0. Reads only level0.heights (a level-0 tile: one chunk)
SAIGA_VULKAN_API GrassMaskData prototypeGrassMask(const TerrainTileData& level0, const TerrainLayout& layout,
                                                  std::optional<int32_t> seaLevel, double minNormalY,
                                                  double minAboveSea);

// min / max of the heights (asserted within kTerrainHeightLimit)
SAIGA_VULKAN_API void terrainHeightRange(const std::vector<int32_t>& heights, int32_t& minHeight, int32_t& maxHeight);

}  // namespace Vulkan
}  // namespace Saiga
