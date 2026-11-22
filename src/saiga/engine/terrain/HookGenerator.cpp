/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/HookGenerator.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/content/gen/FlatGenerator.h"
#include "saiga/engine/terrain/TerrainGeneration.h"

#include <utility>

namespace Saiga
{
namespace Engine
{
HookGenerator::HookGenerator(std::string name, TerrainHeightHook hook, std::optional<HookGrassRule> grass,
                             bool integerOnly)
    : generatorName(std::move(name)), hook(hook), grass(grass), integer(integerOnly)
{
    if (!hook) SAIGA_EXIT_ERROR("HookGenerator '" + generatorName + "': no hook");
}

int32_t HookGenerator::height(uint16_t, const WorldPlan& plan, int64_t x, int64_t z) const
{
    return clampTerrainHeight(hook(plan.seed, x, z));
}

void HookGenerator::grassMask(uint16_t, const WorldPlan& plan, ChunkId chunk, GrassMask& out) const
{
    if (!grass) return zeroGrassMask(plan, out);
    Vulkan::TerrainLayout layout;
    layout.baseSpacing = plan.space.baseSpacing;
    layout.sizeChunks  = plan.space.sizeChunks;
    RegionId region;
    region.space                       = chunk.space;
    region.level                       = 0;
    region.x                           = chunk.x;
    region.z                           = chunk.z;
    const Vulkan::TerrainTileData tile = generateTile(TerrainHeightSource::fromHook(hook, plan.seed), layout, region);
    const Vulkan::GrassMaskData mask =
        generateGrassMask(tile, layout, plan.space.seaLevel, grass->minNormalY, grass->minAboveSea);
    out.size    = mask.size;
    out.density = mask.density ? *mask.density : std::vector<uint8_t>(size_t(mask.size) * size_t(mask.size), 0);
}

}  // namespace Engine
}  // namespace Saiga
