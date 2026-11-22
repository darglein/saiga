/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <vector>

#include <unordered_map>

namespace Saiga
{
namespace Vulkan
{
/**
 * The local lights of RenderScene (S36b, B2 of docs/stages/S36_lighting.plan.md; docs/engine_design.md §10.7): the CPU
 * table of the lights (chunk + chunk-relative position: double-free, R-MATH-1), added one by one (entity lights of the
 * extract) or as one list per chunk (S30's piece lights, never entities). Handles are not reused.
 */
class SAIGA_VULKAN_API LocalLights
{
   public:
    struct Light
    {
        LightHandle handle;
        ChunkId chunk;
        LightData data;
        // The list the light belongs to (0: a single light)
        uint32_t list = 0;
        // The world position (chunk origin + position, double)
        Vec3 worldPosition() const { return chunkOrigin(chunk) + data.position.cast<double>(); }
    };

    LightHandle add(const ChunkId& chunk, const LightData& light);
    // Asserts a live handle that is not part of a list
    void set(LightHandle h, const ChunkId& chunk, const LightData& light);
    void remove(LightHandle h);
    LightListHandle addList(const ChunkId& chunk, const std::vector<LightData>& lights);
    void removeList(LightListHandle list);

    // The live lights (unordered: a removal moves the last light into the gap)
    const std::vector<Light>& lights() const { return table; }
    const Light* find(LightHandle h) const;
    size_t size() const { return table.size(); }
    size_t listCount() const { return lists.size(); }
    // Changes with every add, set and remove (the shadow atlas's cache keys follow the lights themselves)
    uint64_t version() const { return changes; }

   private:
    std::vector<Light> table;
    std::unordered_map<uint32_t, size_t> indexOf;
    std::unordered_map<uint32_t, std::vector<LightHandle>> lists;
    uint32_t nextHandle = 1, nextList = 1;
    uint64_t changes = 0;
    void erase(LightHandle h);
};

}  // namespace Vulkan
}  // namespace Saiga
