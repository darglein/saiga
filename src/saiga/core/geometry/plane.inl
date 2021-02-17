/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "plane.h"

namespace Saiga
{
inline HD Plane::Plane() : normal(vec3(0, 1, 0)), d(0)
{
    static_assert(sizeof(Plane) == 4 * sizeof(float), "Wrong plane size!");
}

inline HD Plane::Plane(const vec3& point, const vec3& normal)
{
    this->normal = normalize(normal);
    d            = dot(point, this->normal);
}

inline HD Plane::Plane(const vec3& p1, const vec3& p2, const vec3& p3)
{
    normal = cross(p2 - p1, p3 - p1);
    normal = normalize(normal);
    d      = dot(p1, this->normal);
}

inline HD Plane Plane::invert() const
{
    return Plane(getPoint(), -normal);
}

inline HD vec3 Plane::closestPointOnPlane(const vec3& p) const
{
    float dis = distance(p);
    return p - dis * normal;
}

inline HD vec3 Plane::getPoint() const
{
    return normal * d;
}


inline HD float Plane::distance(const vec3& p) const
{
    return dot(p, normal) - d;
}

inline HD float Plane::sphereOverlap(const vec3& c, float r) const
{
    return r - distance(c);
}

inline HD vec4 Plane::intersectingCircle(const vec3& c, float r) const
{
    float d      = distance(c);
    float radius = sqrt(r * r - d * d);
    vec3 center  = c + d * normal;
    return make_vec4(center, radius);
}

}  // namespace Saiga
