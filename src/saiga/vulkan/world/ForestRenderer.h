/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/world/FoliageGenerator.h"
#include "saiga/vulkan/world/PrototypeTerrain.h"

#include <vector>

namespace Saiga
{
namespace Vulkan
{
/**
 * The placement of the prototype's trees (S01; S19a: the culling and the draws moved to RenderScene's instance sets,
 * PrototypeForest). From S23 the chunk pipeline places the content.
 */
class SAIGA_VULKAN_API ForestRenderer
{
   public:
    static constexpr int lodCount = 3;

    struct TreeInstance
    {
        // xyz: position of the stem base, w: rotation around y (radians)
        vec4 posYaw;
        // x: height (scale of the unit mesh), y: species, zw: unused
        vec4 scaleSpecies;
    };

    // S20: places count trees in the area on the prototype hook (deterministic for the seeds): the probabilities from
    // a 4 m grid of hook samples over the area (OpenMP, the accepted §3 deviation until S23); a tree stands at
    // float(prototypeSurfaceHeight(area.seed, area.baseSpacing, x, z)) - 0.25 (sunk a little into slopes)
    static std::vector<TreeInstance> placeTrees(const PrototypeArea& area, int count, uint32_t seed,
                                                const std::vector<TreeSpecies>& species);
};

}  // namespace Vulkan
}  // namespace Saiga
