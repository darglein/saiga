/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "aabb.h"
#include <algorithm>
#include <cmath>

namespace Saiga
{
inline float AABB::DistanceSquared(const vec3& p) const
{
    // Distance along each axis
    float dx = std::max({min.x() - p.x(), 0.f, p.x() - max.x()});
    float dy = std::max({min.y() - p.y(), 0.f, p.y() - max.y()});
    float dz = std::max({min.z() - p.z(), 0.f, p.z() - max.z()});
    // total distance squared
    return dx * dx + dy * dy + dz * dz;
}


inline void AABB::transform(const mat4& trafo)
{
    // Bounding box of the transformed box for an affine trafo (Arvo): the center is transformed, the half extent is
    // mapped by the element-wise absolute value of the linear part.
    vec3 center = 0.5f * (min + max);
    vec3 half   = 0.5f * (max - min);
    vec3 new_center, new_half;
    for (int i = 0; i < 3; ++i)
    {
        new_center(i) = trafo(i, 3);
        new_half(i)   = 0;
        for (int j = 0; j < 3; ++j)
        {
            new_center(i) += trafo(i, j) * center(j);
            new_half(i) += std::abs(trafo(i, j)) * half(j);
        }
    }
    min = new_center - new_half;
    max = new_center + new_half;
}

inline void AABB::makeNegative()
{
    const float largeNumber = 100000000000000.0f;
    min                     = make_vec3(largeNumber);
    max                     = make_vec3(-largeNumber);
}

inline void AABB::translate(const vec3& v)
{
    min += v;
    max += v;
}

inline void AABB::scale(const vec3& s)
{
    vec3 pos = getPosition();
    setPosition(make_vec3(0));
    //    min *= s;
    //    max *= s;

    min = min.array() * s.array();
    max = max.array() * s.array();

    setPosition(pos);
}

inline vec3 AABB::getPosition() const
{
    return 0.5f * (min + max);
}

inline void AABB::setPosition(const vec3& v)
{
    vec3 mid = 0.5f * (min + max);
    mid      = v - mid;
    translate(mid);
}
inline std::pair<vec3, float> AABB::BoundingSphere() const
{
    vec3 center  = getPosition();
    vec3 diag    = max - min;
    float radius = diag.norm() * 0.5f;
    return {center, radius};
}


}  // namespace Saiga
