/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The sun and moon shadows of S36a2 on RenderScene (docs/stages/S36_lighting.plan.md, Tests): the multi-view culling
// of the shadow cascades against a CPU reference. One scene per test suite (RenderSceneSuiteTest), skipped without a
// Vulkan 1.3 device.

#include "saiga/vulkan/scene/RenderScene.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "render_scene_test_context.h"

using namespace Saiga;
using namespace Saiga::Vulkan;

namespace
{
// A box of the half extents in one LOD per entry of ends; LOD 0 has all six faces (12 triangles), the others only the
// top and bottom faces (4 triangles), so the triangle statistics show the LOD
MeshLods boxLods(const vec3& half, const std::vector<float>& ends)
{
    struct Face
    {
        vec3 n;
        std::array<vec3, 4> c;
    };
    const std::array<Face, 6> faces = {
        Face{vec3(0, 1, 0), {vec3(-1, 1, 1), vec3(1, 1, 1), vec3(1, 1, -1), vec3(-1, 1, -1)}},
        Face{vec3(0, -1, 0), {vec3(-1, -1, -1), vec3(1, -1, -1), vec3(1, -1, 1), vec3(-1, -1, 1)}},
        Face{vec3(1, 0, 0), {vec3(1, -1, 1), vec3(1, -1, -1), vec3(1, 1, -1), vec3(1, 1, 1)}},
        Face{vec3(-1, 0, 0), {vec3(-1, -1, -1), vec3(-1, -1, 1), vec3(-1, 1, 1), vec3(-1, 1, -1)}},
        Face{vec3(0, 0, 1), {vec3(-1, -1, 1), vec3(1, -1, 1), vec3(1, 1, 1), vec3(-1, 1, 1)}},
        Face{vec3(0, 0, -1), {vec3(1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1), vec3(1, 1, -1)}}};
    MeshLods m;
    for (size_t l = 0; l < ends.size(); ++l)
    {
        MeshLod lod;
        lod.endDistance = ends[l];
        for (size_t fi = 0; fi < (l == 0 ? 6 : 2); ++fi)
        {
            const Face& f       = faces[fi];
            const uint32_t base = uint32_t(lod.vertices.size());
            for (const vec3& c : f.c)
            {
                MeshVertex v;
                v.position = c.cwiseProduct(half);
                v.normal   = packNormalOct(f.n);
                v.color    = packColor(ucvec4(200, 200, 200, 255));
                lod.vertices.push_back(v);
            }
            for (uint32_t i : {0u, 1u, 2u, 0u, 2u, 3u}) lod.indices.push_back(base + i);
        }
        m.lods.push_back(lod);
    }
    m.sphereRadius = half.norm();
    return m;
}

StaticInstance instanceAt(const vec3& position, MeshId mesh, float scale = 1)
{
    StaticInstance s;
    s.setPosition(position);
    s.scale = scale;
    s.mesh  = mesh;
    return s;
}

// A light-space basis for the direction towards the light (the rule of CascadedShadows: x = cross(north, z))
void lightBasis(const Vec3& towardsLight, Vec3& x, Vec3& y, Vec3& z)
{
    z          = towardsLight.normalized();
    Vec3 north = Vec3(0, 0, -1);
    if (std::abs(north.dot(z)) > 0.999) north = Vec3(1, 0, 0);
    x = north.cross(z).normalized();
    y = z.cross(x);
}

// A shadow view: the square of half extent h around the camera-relative point c in light space, the depth range
// [zMin, zMax] and the instances' cap
ShadowView squareView(const Vec3& towardsLight, const Vec3& c, double h, double zMin, double zMax, double cap)
{
    ShadowView v;
    lightBasis(towardsLight, v.axisX, v.axisY, v.axisZ);
    const Vec2 cl(v.axisX.dot(c), v.axisY.dot(c));
    v.rectMin     = cl - Vec2(h, h);
    v.rectMax     = cl + Vec2(h, h);
    v.zMin        = zMin;
    v.zMax        = zMax;
    v.instanceCap = cap;
    v.terrainCap  = cap;
    return v;
}

// The previous view's rectangle shrunk by the band as the de-duplication rectangle of v
void dedupAgainst(ShadowView& v, const ShadowView& previous, double band)
{
    v.dedup    = true;
    v.dedupMin = previous.rectMin + Vec2(band, band);
    v.dedupMax = previous.rectMax - Vec2(band, band);
}

// The CPU reference of a shadow view's count rule (instance_count.comp): the main camera's last end distance, the
// volume, the small-caster rule and the de-duplication; the LOD of the camera distance plus the bias. ambiguous counts
// the instances within 1e-3 (relative) of a decision
struct ViewReference
{
    int count = 0, triangles = 0, ambiguous = 0;
    void add(const Vec3& c, double r, const std::vector<float>& ends, const ShadowView& v)
    {
        const double d = c.norm();
        bool amb       = false;
        for (float e : ends) amb = amb || std::abs(d - e) <= 1e-3 * e;
        bool inside = d <= ends.back();
        for (const Vec4& p : v.planes(false))
        {
            const double s = p(0) * c(0) + p(1) * c(1) + p(2) * c(2) + p(3) + r;
            amb            = amb || std::abs(s) <= 1e-3 * std::max(d, 1.0);
            inside         = inside && s >= 0;
        }
        amb    = amb || std::abs(r - v.smallRadius) <= 1e-4;
        inside = inside && r >= v.smallRadius;
        if (v.dedup)
        {
            const double x = v.axisX.dot(c), y = v.axisY.dot(c);
            const double m =
                std::min({x - r - v.dedupMin(0), v.dedupMax(0) - x - r, y - r - v.dedupMin(1), v.dedupMax(1) - y - r});
            amb = amb || std::abs(m) <= 1e-3 * std::max(d, 1.0);
            if (m >= 0) inside = false;
        }
        if (amb) ambiguous++;
        if (!inside) return;
        int lod = 0;
        while (lod + 1 < int(ends.size()) && d > ends[size_t(lod)]) lod++;
        lod = std::min(lod + v.lodBias, int(ends.size()) - 1);
        count++;
        triangles += lod == 0 ? 12 : 4;
    }
};

class Cascades : public RenderSceneSuiteTest
{
   protected:
    // Renders frames and returns the last frame's statistics
    RenderStatistics Stats(int frames = 2)
    {
        Capture(frames);
        return scene->statistics();
    }
};
}  // namespace

// The culling of shadow views against the CPU reference: 3000 seeded boxes (2 LODs) around the camera, three nested
// light-space squares with de-duplication, the small-caster rule and a LOD bias in the last; per view the counted
// instances and triangles match within the ambiguous ones
TEST_F(Cascades, CullingCountsPerViewMatchCpuReference)
{
    const std::vector<float> ends = {120, 400};
    scene->addMaterial(201, "shadow_cull_flat", MaterialData());
    scene->addMesh(201, "shadow_cull_box", boxLods(vec3(1, 2, 1), ends), 201);
    const Vec3 eye(10.5, 30, -20.25);
    SetCamera(0, eye, Quat::Identity(), 1.0);

    std::mt19937 rng(36);
    std::uniform_real_distribution<float> u(0, 1);
    std::vector<StaticInstance> instances;
    for (int i = 0; i < 3000; ++i)
    {
        const Vec3 world = eye + Vec3((u(rng) - 0.5) * 700, -30 + u(rng) * 20, (u(rng) - 0.5) * 700);
        instances.push_back(instanceAt(world.cast<float>(), 201, 0.2f + 2.0f * u(rng)));
    }
    AddSet(instances);

    const Vec3 light = Vec3(0.4, 0.6, 0.3).normalized();
    std::vector<ShadowView> views;
    views.push_back(squareView(light, Vec3(0, 0, -30), 40, -500, 500, 150));
    views.push_back(squareView(light, Vec3(0, 0, -80), 120, -500, 500, 150));
    dedupAgainst(views[1], views[0], 4);
    views.push_back(squareView(light, Vec3(0, 0, -150), 300, -500, 500, 150));
    dedupAgainst(views[2], views[1], 10);
    views[2].smallRadius = 1.5;
    views[2].lodBias     = 1;
    scene->setShadowViewsOverride(views);

    const RenderStatistics st = Stats();
    for (size_t v = 0; v < views.size(); ++v)
    {
        ViewReference ref;
        for (const StaticInstance& i : instances)
        {
            const Vec3 c   = i.getPosition().cast<double>() - eye;
            const double r = double(std::sqrt(6.0f)) * i.scale;
            ref.add(c, r, ends, views[v]);
        }
        EXPECT_GT(ref.count, 0) << "view " << v + 1;
        EXPECT_LE(std::abs(int(st.cascadeInstances[v]) - ref.count), ref.ambiguous)
            << "view " << v + 1 << ": gpu " << st.cascadeInstances[v] << " reference " << ref.count;
        EXPECT_LE(std::abs(int(st.cascadeTriangles[v]) - ref.triangles), 12 * ref.ambiguous)
            << "view " << v + 1 << ": gpu " << st.cascadeTriangles[v] << " reference " << ref.triangles;
    }
    EXPECT_EQ(st.cascadeInstances[3], 0u);
    scene->setShadowViewsOverride(std::nullopt);
}

// The main view's counts (instancesPerLod, the non-empty buckets) are the same with and without shadow views
TEST_F(Cascades, MainViewCountsUnchanged)
{
    const std::vector<float> ends = {60, 300};
    scene->addMaterial(202, "shadow_main_flat", MaterialData());
    scene->addMesh(202, "shadow_main_box", boxLods(vec3(1, 1, 1), ends), 202);
    const Vec3 eye(0, 10, 0);
    SetCamera(0, eye, Quat::Identity(), 1.0);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(0, 1);
    std::vector<StaticInstance> instances;
    for (int i = 0; i < 2000; ++i)
        instances.push_back(
            instanceAt(vec3((u(rng) - 0.5f) * 400, u(rng) * 5, (u(rng) - 0.5f) * 400), 202, 0.5f + u(rng)));
    AddSet(instances);

    const RenderStatistics without = Stats();
    const Vec3 light               = Vec3(-0.3, 0.8, 0.2).normalized();
    std::vector<ShadowView> views  = {squareView(light, Vec3(0, -10, -20), 50, -400, 400, 100),
                                      squareView(light, Vec3(0, -10, -60), 150, -400, 400, 100)};
    dedupAgainst(views[1], views[0], 5);
    scene->setShadowViewsOverride(views);
    const RenderStatistics with = Stats();
    scene->setShadowViewsOverride(std::nullopt);
    for (int l = 0; l < 4; ++l) EXPECT_EQ(with.instancesPerLod[size_t(l)], without.instancesPerLod[size_t(l)]) << l;
    EXPECT_EQ(with.nonEmptyBuckets, without.nonEmptyBuckets);
    EXPECT_GT(with.cascadeInstances[0], 0u);
    EXPECT_EQ(without.cascadeInstances[0], 0u);
}

// A caster whose xy footprint lies inside the previous cascade's shrunk rectangle is drawn only into that cascade; one
// across the shrunk rectangle's border into both
TEST_F(Cascades, NearCastersNotDuplicated)
{
    scene->addMaterial(203, "shadow_dedup_flat", MaterialData());
    scene->addMesh(203, "shadow_dedup_box", boxLods(vec3(1, 1, 1), {500}), 203);
    const Vec3 eye(0, 5, 0);
    SetCamera(0, eye, Quat::Identity(), 1.0);
    const Vec3 light              = Vec3(0, 1, 0);
    std::vector<ShadowView> views = {squareView(light, Vec3(0, 0, 0), 30, -100, 100, 50),
                                     squareView(light, Vec3(0, 0, 0), 100, -100, 100, 50)};
    dedupAgainst(views[1], views[0], 3);
    scene->setShadowViewsOverride(views);
    // Inside the shrunk rectangle (|x| <= 27): only cascade 1; at x = 27 with radius sqrt(3) across its border: both
    AddSet({instanceAt(vec3(0, 0, -10), 203)});
    RenderStatistics st = Stats();
    EXPECT_EQ(st.cascadeInstances[0], 1u);
    EXPECT_EQ(st.cascadeInstances[1], 0u);
    AddSet({instanceAt(vec3(27, 0, -10), 203)});
    st = Stats();
    EXPECT_EQ(st.cascadeInstances[0], 2u);
    EXPECT_EQ(st.cascadeInstances[1], 1u);
    scene->setShadowViewsOverride(std::nullopt);
}

// The small-caster rule: a caster whose radius is below the view's limit is skipped there and drawn where the limit is
// lower
TEST_F(Cascades, SmallCastersSkippedInFarCascades)
{
    scene->addMaterial(204, "shadow_small_flat", MaterialData());
    scene->addMesh(204, "shadow_small_box", boxLods(vec3(1, 1, 1), {500}), 204);
    SetCamera(0, Vec3(0, 5, 0), Quat::Identity(), 1.0);
    const Vec3 light              = Vec3(0.2, 1, 0.1).normalized();
    std::vector<ShadowView> views = {squareView(light, Vec3(0, 0, 0), 200, -100, 100, 50),
                                     squareView(light, Vec3(0, 0, 0), 400, -100, 100, 50)};
    views[1].smallRadius          = 1.0;
    scene->setShadowViewsOverride(views);
    // Radius sqrt(3) * 0.3 = 0.52 (small) and sqrt(3) * 2 = 3.5
    AddSet({instanceAt(vec3(5, 0, -20), 204, 0.3f), instanceAt(vec3(-5, 0, -20), 204, 2.0f)});
    const RenderStatistics st = Stats();
    EXPECT_EQ(st.cascadeInstances[0], 2u);
    EXPECT_EQ(st.cascadeInstances[1], 1u);
    scene->setShadowViewsOverride(std::nullopt);
}

// A mesh beyond its last end distance from the main camera casts nothing (finding 20d), also where a shadow view
// contains it
TEST_F(Cascades, ShadowViewsKeepMainEndDistance)
{
    scene->addMaterial(205, "shadow_end_flat", MaterialData());
    scene->addMesh(205, "shadow_end_box", boxLods(vec3(1, 1, 1), {100, 400}), 205);
    // Looking along -z: the boxes behind the camera are outside the main view but inside the shadow view
    SetCamera(0, Vec3(0, 5, 0), Quat::Identity(), 1.0);
    std::vector<ShadowView> views = {squareView(Vec3(0, 1, 0), Vec3(0, 0, 0), 1000, -100, 100, 50)};
    scene->setShadowViewsOverride(views);
    AddSet({instanceAt(vec3(0, 0, 300), 205), instanceAt(vec3(0, 0, 500), 205)});
    const RenderStatistics st = Stats();
    EXPECT_EQ(st.cascadeInstances[0], 1u);
    EXPECT_EQ(st.instancesPerLod[0] + st.instancesPerLod[1], 0u);
    scene->setShadowViewsOverride(std::nullopt);
}
