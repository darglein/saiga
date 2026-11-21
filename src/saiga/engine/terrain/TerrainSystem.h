/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainSystem (docs/engine_design.md §12.2, §12.3; S20): the terrain of the World's spaces. The exact surface
// height (World::terrainHeight) and the base-grid samples of a space's height source.

#pragma once

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/world/SpaceTemplate.h"
#include "saiga/export.h"

#include <vector>

namespace Saiga
{
namespace Engine
{
class SAIGA_ENGINE_API TerrainSystem
{
   public:
    // One entry per space (SpaceId = index): the template and the space seed
    TerrainSystem(const std::vector<SpaceTemplate>& templates, const std::vector<uint64_t>& seeds);
    ~TerrainSystem();
    TerrainSystem(const TerrainSystem&)            = delete;
    TerrainSystem& operator=(const TerrainSystem&) = delete;

    // World::terrainHeight (§7.10, §12.2): x and z clamped to the space [0, sizeChunks * 256]; the cell of the base
    // grid (the last cell at the far edge), the three samples of its triangle (sample) and terrainSurfaceDouble, in
    // metres. 0 for a space without terrain. Const and thread safe (parallelEach)
    double height(SpaceId space, double x, double z) const;
    // A base-grid sample of the BASE data: the space's height source at (sx, sz) * baseSpacing (S20 step 6: a resident
    // level-0 tile when there is one); 0 for a space without terrain. A cheap point query, const and thread safe
    // (parallelEach). S22's World::terrainSample = this + the edit delta; S23a replaces only the source
    int32_t sample(SpaceId space, int64_t sx, int64_t sz) const;

    const SpaceTemplate& spaceTemplate(SpaceId space) const;
    const TerrainHeightSource& heightSource(SpaceId space) const;
    uint64_t spaceSeed(SpaceId space) const { return spaces[space].seed; }
    size_t spaceCount() const { return spaces.size(); }

   private:
    struct Space
    {
        SpaceTemplate settings;
        uint64_t seed = 0;
        TerrainHeightSource source;
    };
    std::vector<Space> spaces;
};

}  // namespace Engine
}  // namespace Saiga
