/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SitePatch (docs/engine_design.md §12.1 (a), amendment 4 of S23a): the engine composes the sites of a plan into the
// natural terrain and applies their exclusions. A site's content in its local frame (integer metres) as plain integer
// data (the pipeline converts each prefab once; content code never includes the prefab header), the quarter-turn
// rotations and the composed height.
//
// Quarter turns (fixed: they enter the committed hashes): a site's rotation is quarterTurns * 90 degrees about +y in
// the yaw direction; a local point (x, z) maps to q = 1: (z, -x), q = 2: (-x, -z), q = 3: (-z, x), then the site
// origin is added. A point-in-area test inverse-rotates the point (by 4 - q) into the local frame and tests it against
// the local half-open rectangle; a rotated area uses the closed integer bounds of the rectangle's corners.

#pragma once

#include "saiga/engine/content/WorldPlan.h"
#include "saiga/engine/content/gen/FixedMath.h"
#include "saiga/export.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Saiga
{
namespace Engine
{
class Generator;

// A prefab's terrain patch: samples 1 m apart at [originX, originX + sizeX) x [originZ, originZ + sizeZ) (local
// metres), heights relative to the site's baseHeight, weights 0..256
struct PatchData
{
    int32_t originX = 0, originZ = 0, sizeX = 0, sizeZ = 0;
    std::vector<int32_t> heights, weights;
};

// What the pipeline composes of a site's prefab (local frame)
struct SiteContent
{
    std::shared_ptr<const PatchData> patch;  // null: no terrain patch
    std::vector<int32_t> exclusion;          // 4 values per rectangle [x0, z0, x1, z1), integer metres
};

// The rotation of a local point by q quarter turns (q in 0..3)
constexpr void rotateQuarter(int64_t x, int64_t z, uint32_t q, int64_t& outX, int64_t& outZ)
{
    switch (q & 3)
    {
        case 0:
            outX = x;
            outZ = z;
            break;
        case 1:
            outX = z;
            outZ = -x;
            break;
        case 2:
            outX = -x;
            outZ = -z;
            break;
        default:
            outX = -z;
            outZ = x;
            break;
    }
}
// The inverse rotation (by 4 - q)
constexpr void inverseRotateQuarter(int64_t x, int64_t z, uint32_t q, int64_t& outX, int64_t& outZ)
{
    rotateQuarter(x, z, (4 - (q & 3)) & 3, outX, outZ);
}
// The closed integer bounds [x0, x1] x [z0, z1] of a local rectangle rotated by q (corners rotated, then min / max);
// in the site's local frame, without the origin
SAIGA_ENGINE_API void rotateBounds(const int32_t rect[4], uint32_t q, int64_t out[4]);
// A point (any unit, relative to the site origin in the same unit) inside a local half-open rectangle given in metres
// times unitsPerMetre
SAIGA_ENGINE_API bool insideLocalRect(int64_t dx, int64_t dz, uint32_t q, const int32_t rect[4], int64_t unitsPerMetre);

// The patched height at an integer point (x, z) in metres of the space: natural outside the site's patch, inside
// natural + floor((site.baseHeight + patch - natural) * w / 256) (int64). False if the point is outside the patch
SAIGA_ENGINE_API bool patchedHeight(const Site& site, const PatchData& patch, int64_t x, int64_t z, int32_t natural,
                                    int32_t& out);

// The composed height of a space: the generator's natural height with every patch of the sites listed in the point's
// chunk applied in site order (plan.sitesByChunk, space = the plan's SpaceId). Checked (every build) to lie in
// +-kTerrainHeightLimit. Const, lock-free, allocation-free
SAIGA_ENGINE_API int32_t composedHeight(const Generator& generator, uint16_t version, const WorldPlan& plan,
                                        SpaceId space, const std::vector<std::shared_ptr<const SiteContent>>& sites,
                                        int64_t x, int64_t z);
// A point in 1/256 m of the space inside an exclusion rectangle of a site listed in the point's chunk
SAIGA_ENGINE_API bool excluded(const WorldPlan& plan, SpaceId space,
                               const std::vector<std::shared_ptr<const SiteContent>>& sites, int64_t x256,
                               int64_t z256);

}  // namespace Engine
}  // namespace Saiga
