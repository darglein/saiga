/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

#include <algorithm>
#include <array>

namespace Saiga
{
namespace Vulkan
{
// The camera rotation (RenderScene::setCamera: looks along -z, y up) of a view matrix (rotation and translation)
inline Quat viewRotation(const mat4& view)
{
    Mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r(i, j) = double(view(j, i));
    return Quat(r).normalized();
}

// Frustum planes of a (reversed, infinite) projection: left, right, bottom, top, near (normalized). The 6th plane is
// a dummy that is always passed (0,0,0,1).
inline std::array<vec4, 6> frustumPlanes(const mat4& viewProj)
{
    auto row = [&](int r) { return vec4(viewProj(r, 0), viewProj(r, 1), viewProj(r, 2), viewProj(r, 3)); };
    vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    std::array<vec4, 6> p = {r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2, vec4(0, 0, 0, 1)};
    for (int i = 0; i < 5; ++i)
    {
        float l = make_vec3(p[i]).norm();
        p[i] /= l;
    }
    return p;
}

// True if the axis aligned box is (partly) inside the frustum planes.
inline bool boxInFrustum(const std::array<vec4, 6>& planes, const vec3& bmin, const vec3& bmax)
{
    for (auto& p : planes)
    {
        // The corner furthest in the direction of the plane normal
        vec3 v(p(0) >= 0 ? bmax(0) : bmin(0), p(1) >= 0 ? bmax(1) : bmin(1), p(2) >= 0 ? bmax(2) : bmin(2));
        if (p(0) * v(0) + p(1) * v(1) + p(2) * v(2) + p(3) < 0) return false;
    }
    return true;
}

// Squared distance from the point to the axis aligned box (0 inside).
inline float boxDistanceSquared(const vec3& p, const vec3& bmin, const vec3& bmax)
{
    float d = 0;
    for (int i = 0; i < 3; ++i)
    {
        float v = std::max({bmin(i) - p(i), 0.0f, p(i) - bmax(i)});
        d += v * v;
    }
    return d;
}

}  // namespace Vulkan
}  // namespace Saiga
