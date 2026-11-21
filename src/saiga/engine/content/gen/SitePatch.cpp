/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/gen/SitePatch.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/content/Generator.h"

#include <string>

namespace Saiga
{
namespace Engine
{
void rotateBounds(const int32_t rect[4], uint32_t q, int64_t out[4])
{
    int64_t ax = 0, az = 0, bx = 0, bz = 0;
    rotateQuarter(rect[0], rect[1], q, ax, az);
    rotateQuarter(rect[2], rect[3], q, bx, bz);
    out[0] = ax < bx ? ax : bx;
    out[1] = az < bz ? az : bz;
    out[2] = ax < bx ? bx : ax;
    out[3] = az < bz ? bz : az;
}

bool insideLocalRect(int64_t dx, int64_t dz, uint32_t q, const int32_t rect[4], int64_t unitsPerMetre)
{
    int64_t lx = 0, lz = 0;
    inverseRotateQuarter(dx, dz, q, lx, lz);
    return lx >= int64_t(rect[0]) * unitsPerMetre && lx < int64_t(rect[2]) * unitsPerMetre &&
           lz >= int64_t(rect[1]) * unitsPerMetre && lz < int64_t(rect[3]) * unitsPerMetre;
}

bool patchedHeight(const Site& site, const PatchData& patch, int64_t x, int64_t z, int32_t natural, int32_t& out)
{
    int64_t lx = 0, lz = 0;
    inverseRotateQuarter(x - site.x, z - site.z, site.quarterTurns, lx, lz);
    const int64_t i = lx - patch.originX, k = lz - patch.originZ;
    if (i < 0 || k < 0 || i >= patch.sizeX || k >= patch.sizeZ) return false;
    const uint32_t index = uint32_t(k * patch.sizeX + i);
    const int64_t target = int64_t(site.baseHeight) + patch.heights[index];
    out = int32_t(int64_t(natural) + floorShift((target - natural) * int64_t(patch.weights[index]), 8));
    return true;
}

int32_t composedHeight(const Generator& generator, uint16_t version, const WorldPlan& plan, SpaceId space,
                       const std::vector<std::shared_ptr<const SiteContent>>& sites, int64_t x, int64_t z)
{
    int32_t h = generator.height(version, plan, x, z);
    if (!plan.sitesByChunk.empty())
    {
        ChunkId c;
        c.space = space;
        c.x     = int32_t(floorShift(x, 8));
        c.z     = int32_t(floorShift(z, 8));
        auto it = plan.sitesByChunk.find(c);
        if (it != plan.sitesByChunk.end())
            for (uint32_t s : it->second)
            {
                const SiteContent* content = sites[s].get();
                if (!content || !content->patch) continue;
                int32_t patched = 0;
                if (patchedHeight(plan.sites[s], *content->patch, x, z, h, patched)) h = patched;
            }
    }
    if (h > kTerrainHeightLimit || h < -kTerrainHeightLimit)
        SAIGA_EXIT_ERROR(std::string("generator '") + generator.name() + "': the height " + std::to_string(h) +
                         " at (" + std::to_string(x) + ", " + std::to_string(z) + ") lies beyond kTerrainHeightLimit");
    return h;
}

bool excluded(const WorldPlan& plan, SpaceId space, const std::vector<std::shared_ptr<const SiteContent>>& sites,
              int64_t x256, int64_t z256)
{
    if (plan.sitesByChunk.empty()) return false;
    ChunkId c;
    c.space = space;
    c.x     = int32_t(floorShift(x256, 16));
    c.z     = int32_t(floorShift(z256, 16));
    auto it = plan.sitesByChunk.find(c);
    if (it == plan.sitesByChunk.end()) return false;
    for (uint32_t s : it->second)
    {
        const SiteContent* content = sites[s].get();
        if (!content) continue;
        const Site& site = plan.sites[s];
        for (uint32_t r = 0; r + 3 < uint32_t(content->exclusion.size()); r += 4)
            if (insideLocalRect(x256 - site.x * kPosUnitsPerMetre, z256 - site.z * kPosUnitsPerMetre, site.quarterTurns,
                                &content->exclusion[r], kPosUnitsPerMetre))
                return true;
    }
    return false;
}

}  // namespace Engine
}  // namespace Saiga
