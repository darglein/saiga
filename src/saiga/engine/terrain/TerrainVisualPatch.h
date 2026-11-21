/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainVisualPatch (docs/engine_design.md §12.2; S22): the terrain edits on the visual path, owned by TerrainSystem.
// The base data in the TerrainStore is never mutated (D3): a tile or the pyramid that reaches the sink is merged at the
// send point (a copy with clampTerrainHeight(base + delta) at the samples that coincide with allocated delta blocks and
// exact min / max; the shared base object when no chunk it covers has deltas), and the changed boxes of later edits are
// patched with mergedHeights (the base from the store's own data plus the deltas; no height source call). The edited
// grass masks (S22 step 9) are cached here too.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/terrain/SampleRect.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <cstdint>
#include <vector>

namespace Saiga
{
namespace Engine
{
class TerrainEditStore;

class SAIGA_ENGINE_API TerrainVisualPatch
{
   public:
    // A chunk with deltas lies in the chunk range [x0, x1] x [z0, z1] of the space
    static bool hasRecords(const TerrainEditStore& edits, SpaceId space, int64_t x0, int64_t z0, int64_t x1,
                           int64_t z1);
    // The merged copy of a tile of the given layout; false (out untouched) when no chunk it covers has deltas
    static bool mergeTile(const TerrainEditStore& edits, const RegionId& region, const Vulkan::TerrainLayout& layout,
                          const Vulkan::TerrainTileData& base, Vulkan::TerrainTileData& out);
    // The merged copy of the pyramid (32 m samples of the whole space); false when the space has no deltas
    static bool mergePyramid(const TerrainEditStore& edits, SpaceId space, const Vulkan::TerrainPyramidData& base,
                             Vulkan::TerrainPyramidData& out);
    // w x h merged heights (row-major) of the samples (i0 + a * step, k0 + b * step): base holds the same samples
    // with the row stride baseStride; out = clampTerrainHeight(base + delta)
    static void mergedHeights(const TerrainEditStore& edits, SpaceId space, int64_t i0, int64_t k0, int64_t step, int w,
                              int h, const int32_t* base, size_t baseStride, int32_t* out);
};

}  // namespace Engine
}  // namespace Saiga
