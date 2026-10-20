/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "intersection.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
namespace Intersection
{
bool SphereAABB(const vec3& c, float r, const AABB& bb2)
{
    return bb2.DistanceSquared(c) <= r * r;
}
}  // namespace Intersection
}  // namespace Saiga
