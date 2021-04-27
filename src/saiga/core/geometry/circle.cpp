/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "circle.h"

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
std::ostream& operator<<(std::ostream& os, const Saiga::Circle& s)
{
    os << "Circle: " << s.pos << s.r << s.normal;
    return os;
}

float Circle::distance(const vec3& p) const
{
    return Saiga::distance(p, closestPointOnCircle(p));
}

vec3 Circle::closestPointOnCircle(const vec3& p) const
{
    vec3 pp = getPlane().closestPointOnPlane(p);
    vec3 d  = normalize(vec3(pp - pos));
    return pos + d * r;
}

}  // namespace Saiga
