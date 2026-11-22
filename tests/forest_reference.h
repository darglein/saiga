/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The CPU reference of the prototype trees' culling (§17.5; S20: moved out of test_vulkan_world.cpp, over the placed
// trees, so the goldens in test_engine_terrain_render keep the check)

#pragma once

#include "saiga/vulkan/world/ForestRenderer.h"
#include "saiga/vulkan/world/PrototypeForest.h"
#include "saiga/vulkan/world/WorldSettings.h"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace ForestReferenceTests
{
using namespace Saiga;
using namespace Saiga::Vulkan;

// Per LOD the drawn trees and the ambiguous ones (within 1e-3 relative of a plane or an end distance)
struct ForestReference
{
    std::array<int, 3> perLod{};
    int ambiguous = 0;
    int total() const { return perLod[0] + perLod[1] + perLod[2]; }
};

// The inward camera-relative frustum planes of RenderScene (left, right, bottom, top, near) for the camera
inline std::array<Vec4, 5> cameraPlanes(const Quat& rotation, double fovy, double aspect, double nearPlane)
{
    const Mat3 R      = rotation.matrix();
    const double tanY = std::tan(fovy / 2), tanX = tanY * aspect;
    std::array<Vec4, 5> p;
    const Vec3 n[4] = {Vec3(1, 0, -tanX), Vec3(-1, 0, -tanX), Vec3(0, 1, -tanY), Vec3(0, -1, -tanY)};
    for (int k = 0; k < 4; ++k)
    {
        const Vec3 m = R * n[k].normalized();
        p[k]         = Vec4(m(0), m(1), m(2), 0);
    }
    const Vec3 f = R * Vec3(0, 0, -1);
    p[4]         = Vec4(f(0), f(1), f(2), -nearPlane);
    return p;
}

// The prototype's rule as the instance path implements it, for the settings' fovy, LOD distances, padding and toggle
inline ForestReference forestReference(const std::vector<ForestRenderer::TreeInstance>& trees,
                                       const std::vector<TreeSpecies>& species, const WorldSettings& settings,
                                       const Vec3& eye, const Quat& rotation, double aspect)
{
    ForestReference r;
    const auto ends   = prototypeTreeLodEnds(settings);
    const double pad  = prototypeTreePadding(settings);
    const auto planes = cameraPlanes(rotation, double(radians(settings.fovy)), aspect, 0.1);
    if (!settings.trees) return r;
    for (const auto& t : trees)
    {
        const TreeSpecies& sp = species[size_t(t.scaleSpecies(1))];
        const double scale    = t.scaleSpecies(0);
        const Vec3 center =
            Vec3(t.posYaw(0), t.posYaw(1), t.posYaw(2)) + Vec3(0, double(sp.bounds(0)) * scale, 0) - eye;
        const double radius = (double(sp.bounds(1)) + pad) * scale;
        const double d      = center.norm();
        bool amb            = false;
        for (float e : ends) amb = amb || std::abs(d - e) <= 1e-3 * e;
        bool inside = d <= ends[2];
        for (const Vec4& p : planes)
        {
            const double s = p(0) * center(0) + p(1) * center(1) + p(2) * center(2) + p(3) + radius;
            amb            = amb || std::abs(s) <= 1e-3 * std::max(d, 1.0);
            inside         = inside && s >= 0;
        }
        if (amb) r.ambiguous++;
        if (!inside) continue;
        int lod = 0;
        while (lod < 2 && d > ends[size_t(lod)]) lod++;
        r.perLod[size_t(lod)]++;
    }
    return r;
}

inline void expectForest(const PrototypeForest::Stats& s, const ForestReference& r, const std::string& what)
{
    int total = 0;
    for (int l = 0; l < 3; ++l)
    {
        EXPECT_LE(std::abs(int(s.visible[size_t(l)]) - r.perLod[size_t(l)]), r.ambiguous)
            << what << ": LOD " << l << " gpu " << s.visible[size_t(l)] << " reference " << r.perLod[size_t(l)];
        total += int(s.visible[size_t(l)]);
    }
    EXPECT_LE(std::abs(total - r.total()), r.ambiguous) << what;
}

}  // namespace ForestReferenceTests
