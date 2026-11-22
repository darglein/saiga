/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The sun and moon shadows of S36a2 on RenderScene (docs/stages/S36_lighting.plan.md, Tests): the multi-view culling
// of the shadow cascades against a CPU reference, the cascades' fitting (pure), the rendered cascade shadows (the
// ShadowFactor view read back) and the shadow pipeline table. One scene per test suite (RenderSceneSuiteTest), skipped
// without a Vulkan 1.3 device.

#include "saiga/vulkan/scene/CascadedShadows.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/ShadowPipelineTable.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "render_scene_test_context.h"
#include "terrain_test_hooks.h"

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

//==================================================================================================
// The cascades' fitting (CPU, CascadedShadows::fit; no device needed, the suite's fixture only skips without one)

namespace
{
constexpr double kPi = 3.14159265358979323846;

Quat lookRotation(const Vec3& forward)
{
    const Vec3 back  = -forward.normalized();
    Vec3 up0         = std::abs(back(1)) > 0.999 ? Vec3(0, 0, -1) : Vec3(0, 1, 0);
    const Vec3 right = up0.cross(back).normalized();
    const Vec3 up    = back.cross(right);
    Mat3 R;
    for (int i = 0; i < 3; ++i)
    {
        R(i, 0) = right(i);
        R(i, 1) = up(i);
        R(i, 2) = back(i);
    }
    return Quat(R).normalized();
}

CascadeInput cameraInput(const Vec3& eye, const Vec3& forward, const Vec3& light, uint64_t frame)
{
    CascadeInput in;
    in.camera         = eye;
    in.cameraRotation = lookRotation(forward).matrix();
    in.fovy           = 1.0;
    in.aspect         = 16.0 / 9.0;
    in.nearPlane      = 0.1;
    in.towardsLight   = light.normalized();
    in.frame          = frame;
    in.maxSpeed       = 50;
    in.dt             = 1.0 / 60.0;
    return in;
}

// The NDC of a world point in cascade c (camera-relative through the cascade's matrix of that camera)
Vec3 cascadeNdc(const CascadedShadows& cs, int c, const Vec3& camera, const Vec3& world)
{
    const mat4 m   = cs.matrix(c, camera);
    const vec3 rel = (world - camera).cast<float>();
    const vec4 h   = m * vec4(rel(0), rel(1), rel(2), 1);
    return Vec3(h(0), h(1), h(2));
}
bool insideCascade(const CascadedShadows& cs, int c, const Vec3& camera, const Vec3& world)
{
    const Vec3 n = cascadeNdc(cs, c, camera, world);
    return std::abs(n(0)) <= 1 && std::abs(n(1)) <= 1 && n(2) >= 0 && n(2) <= 1;
}
// A point of the camera frustum at the distance d and the screen position (sx, sy) in [-1, 1]
Vec3 frustumPoint(const CascadeInput& in, double sx, double sy, double d)
{
    const double tanY = std::tan(in.fovy * 0.5), tanX = tanY * in.aspect;
    return in.camera + in.cameraRotation * Vec3(sx * tanX * d, sy * tanY * d, -d);
}
}  // namespace

// The splits are lambda * log + (1 - lambda) * linear between the near plane and the shadow distance
TEST_F(Cascades, SplitsMatchFormula)
{
    const std::vector<double> s = CascadedShadows::cascadeSplits(0.1, 1500, 4, 0.8);
    ASSERT_EQ(s.size(), 5u);
    for (int i = 0; i <= 4; ++i)
    {
        const double t = i / 4.0;
        const double e = 0.8 * 0.1 * std::pow(15000.0, t) + 0.2 * (0.1 + 1499.9 * t);
        EXPECT_NEAR(s[size_t(i)], e, 1e-9 * std::max(1.0, e)) << i;
    }
    EXPECT_DOUBLE_EQ(s.front(), 0.1);
    EXPECT_DOUBLE_EQ(s.back(), 1500);
    for (int i = 0; i < 4; ++i) EXPECT_LT(s[size_t(i)], s[size_t(i) + 1]);
}

// Every point of the camera frustum up to the shadow distance lies inside some cascade's box (xy and depth)
TEST_F(Cascades, CoverCameraFrustum)
{
    LightingParameters p;
    p.sunShadows = true;
    for (const Vec3& light : {Vec3(0.3, 0.8, 0.5), Vec3(-0.9, 0.09, 0.2), Vec3(0, 1, 0)})
    {
        CascadedShadows cs;
        const CascadeInput in = cameraInput(Vec3(100.25, 40, -300.5), Vec3(0.6, -0.2, -1), light, 1);
        cs.fit(in, p, false);
        std::mt19937 rng(5);
        std::uniform_real_distribution<double> u(0, 1);
        int outside = 0;
        for (int i = 0; i < 4000; ++i)
        {
            const double d = in.nearPlane + (p.shadowDistance - in.nearPlane) * u(rng);
            const Vec3 w   = frustumPoint(in, u(rng) * 2 - 1, u(rng) * 2 - 1, d);
            bool any       = false;
            for (int c = 0; c < cs.count(); ++c) any = any || insideCascade(cs, c, in.camera, w);
            outside += any ? 0 : 1;
        }
        EXPECT_EQ(outside, 0) << light.transpose();
    }
}

// Texel snapping: a camera move of 0.3 texel moves a cascade's centre by a whole number of texels (at most one, the
// matrices map world points to the same texel grid); a rotation keeps the sphere radius
TEST_F(Cascades, StableUnderSubTexelMoves)
{
    LightingParameters p;
    p.sunShadows     = true;
    const Vec3 light = Vec3(0.4, 0.7, 0.3);
    CascadedShadows a, b, r;
    CascadeInput in = cameraInput(Vec3(10, 30, 20), Vec3(1, -0.3, -0.5), light, 1);
    a.fit(in, p, false);
    for (int c = 0; c < 3; ++c)
    {
        const auto& ka     = a.cascades()[size_t(c)];
        const double texel = ka.texelSize;
        CascadeInput moved = in;
        moved.camera += ka.axisX * (0.3 * texel) + ka.axisY * (0.3 * texel);
        b.fit(moved, p, false);
        const auto& kb = b.cascades()[size_t(c)];
        EXPECT_DOUBLE_EQ(kb.texelSize, texel);
        for (double d : {(kb.cx - ka.cx) / texel, (kb.cy - ka.cy) / texel})
        {
            EXPECT_NEAR(d, std::round(d), 1e-6) << c;
            EXPECT_LE(std::abs(d), 1.0 + 1e-6) << c;
        }
        CascadeInput turned   = in;
        turned.cameraRotation = lookRotation(Vec3(-0.7, 0.1, 0.4)).matrix();
        r.fit(turned, p, false);
        EXPECT_DOUBLE_EQ(r.cascades()[size_t(c)].halfExtent, ka.halfExtent) << c;
    }
}

// A cascade that is not updated in a frame (interval 2) shadows the same world point after a camera move: its matrix
// is rebuilt camera-relative from the double centre
TEST_F(Cascades, IntervalKeepsMatrixCorrect)
{
    LightingParameters p;
    p.sunShadows = true;
    CascadedShadows cs;
    CascadeInput in = cameraInput(Vec3(1000.5, 50, -2000.25), Vec3(0, -0.2, -1), Vec3(0.2, 0.6, 0.4), 1);
    in.maxSpeed     = 300;  // a margin of 2 * 300 / 60 = 10 m
    cs.fit(in, p, false);
    const Vec3 world(1200, 20, -2600);
    const Vec3 before = cascadeNdc(cs, 3, in.camera, world);
    CascadeInput next = in;
    next.camera += Vec3(3.3, 0.5, -4.1);
    next.frame = 2;
    cs.fit(next, p, false);
    ASSERT_FALSE(cs.cascades()[3].updated);
    EXPECT_TRUE(cs.cascades()[0].updated);
    const Vec3 after = cascadeNdc(cs, 3, next.camera, world);
    EXPECT_NEAR(before(0), after(0), 2e-5);
    EXPECT_NEAR(before(1), after(1), 2e-5);
    EXPECT_NEAR(before(2), after(2), 2e-5);
    next.frame = 3;
    cs.fit(next, p, false);
    EXPECT_TRUE(cs.cascades()[3].updated);
}

// A stale cascade (interval 2) still covers every receiver of its slice after a turn of 30 degrees and a move of 20 m
// between its updates (its sphere around the camera has the margin interval * maxSpeed * dt)
TEST_F(Cascades, StaleCascadeStillCoversFrustum)
{
    LightingParameters p;
    p.sunShadows = true;
    CascadedShadows cs;
    CascadeInput in = cameraInput(Vec3(0, 60, 0), Vec3(0, -0.1, -1), Vec3(-0.3, 0.5, 0.6), 1);
    in.maxSpeed     = 600;
    cs.fit(in, p, false);
    CascadeInput next   = in;
    next.frame          = 2;
    next.camera         = in.camera + Vec3(12, 0, -16);  // 20 m
    next.cameraRotation = lookRotation(Vec3(std::sin(kPi / 6), -0.1, -std::cos(kPi / 6))).matrix();
    cs.fit(next, p, false);
    ASSERT_FALSE(cs.cascades()[3].updated);
    const auto& k = cs.cascades()[3];
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> u(0, 1);
    int outside = 0;
    for (int i = 0; i < 4000; ++i)
    {
        const double d = k.splitNear + (k.splitFar - k.splitNear) * u(rng);
        if (!insideCascade(cs, 3, next.camera, frustumPoint(next, u(rng) * 2 - 1, u(rng) * 2 - 1, d))) outside++;
    }
    EXPECT_EQ(outside, 0);
}

// The light basis near the zenith: orthonormal, and a light tilted by 0.01 rad changes the basis by about as much (no
// flip); towards the north horizon the fallback keeps it finite
TEST_F(Cascades, BasisStableNearZenith)
{
    Vec3 x0, y0, z0;
    CascadedShadows::lightBasis(Vec3(0, 1, 0), x0, y0, z0);
    for (double a = 0; a < 2 * kPi; a += kPi / 8)
    {
        Vec3 x, y, z;
        CascadedShadows::lightBasis(Vec3(0.01 * std::cos(a), 1, 0.01 * std::sin(a)), x, y, z);
        EXPECT_NEAR(x.dot(y), 0, 1e-12);
        EXPECT_NEAR(x.norm(), 1, 1e-12);
        EXPECT_LT((x - x0).norm(), 0.03) << a;
        EXPECT_LT((y - y0).norm(), 0.03) << a;
    }
    for (const Vec3& l : {Vec3(0, 0, -1), Vec3(0, 0.001, -1), Vec3(0, 0, 1)})
    {
        Vec3 x, y, z;
        CascadedShadows::lightBasis(l, x, y, z);
        EXPECT_TRUE(x.allFinite() && y.allFinite());
        EXPECT_NEAR(x.dot(z), 0, 1e-9);
        EXPECT_NEAR(y.norm(), 1, 1e-9);
    }
}

// The shadow light switch (D13): by ground illuminance with a hysteresis of a factor 2, no alternation around the
// threshold; none when neither body is above the horizon
TEST_F(Cascades, ShadowLightSwitchHasHysteresis)
{
    EXPECT_EQ(CascadedShadows::selectShadowLight(0, 0, 0), 0);
    EXPECT_EQ(CascadedShadows::selectShadowLight(0, 10, 1), 1);
    EXPECT_EQ(CascadedShadows::selectShadowLight(0, 0, 0.1), 2);
    // From the sun: the moon takes over only when it is twice as bright on the ground
    int choice = 1;
    for (double sun : {1.0, 0.8, 0.6, 0.5, 0.45, 0.6, 0.5})
    {
        choice = CascadedShadows::selectShadowLight(choice, sun, 1.0);
        EXPECT_EQ(choice, sun >= 0.5 ? 1 : 2) << sun;
        if (choice == 2) break;
    }
    // Around the threshold the moon stays (the sun must be twice as bright to come back)
    for (double sun : {0.6, 1.0, 1.5, 1.99})
        EXPECT_EQ(choice = CascadedShadows::selectShadowLight(choice, sun, 1.0), 2);
    EXPECT_EQ(CascadedShadows::selectShadowLight(choice, 2.01, 1.0), 1);
    // The ground illuminance: 0 below the horizon, rising with the elevation
    EXPECT_EQ(CascadedShadows::groundIlluminance(120000, -0.01), 0);
    EXPECT_LT(CascadedShadows::groundIlluminance(120000, 0.05), CascadedShadows::groundIlluminance(120000, 0.5));
    EXPECT_GT(CascadedShadows::groundIlluminance(120000, kPi / 2), 90000);
}

//==================================================================================================
// The rendered cascade shadows (the ShadowFactor view: the raw factor in the fp16 target, read back as 8 bit)

namespace
{
class Shadows : public RenderSceneSuiteTest
{
   protected:
    ivec2 Size() const override { return ivec2(256, 256); }
    int Samples() const override { return 1; }

    void SetUp() override
    {
        RenderSceneSuiteTest::SetUp();
        if (!scene) return;
        LightingParameters p;
        p.sunShadows        = true;
        p.cascadeResolution = 1024;
        p.shadowDistance    = 200;
        scene->setLightingParameters(p);
        scene->setLightingDebug({LightingDebugView::ShadowFactor, 0});
    }
    void TearDown() override
    {
        if (scene)
        {
            scene->setLightingDebug(LightingDebugState());
            scene->setFreezeCascades(false);
            for (int c = 0; c < pipelineClassCount; ++c)
                scene->shadowPipelines().setEntry(c, ShadowPipelineTable::defaultEntry(c));
        }
        RenderSceneSuiteTest::TearDown();
    }

    // The studio sun towards the light (clay: the shadow light of the cascades), no fog
    void Sun(const Vec3& towardsLight)
    {
        StudioEnvironment s;
        s.sunDir = towardsLight.normalized().cast<float>();
        s.fog    = false;
        scene->setStudioEnvironment(s);
    }
    // A mesh of a box (one LOD to far away)
    void Box(MeshId id, const vec3& half)
    {
        MeshLods m = boxLods(half, {1e6f});
        scene->addMaterial(MaterialId(id), "shadow_box_" + std::to_string(id), MaterialData());
        scene->addMesh(id, "shadow_box_" + std::to_string(id), m, MaterialId(id));
    }
    // The factor (0..1) at a world point of the last frame (its matrices), -1 outside the image
    float FactorAt(const TemplatedImage<ucvec4>& img, const Vec3& world)
    {
        const Vec3 rel = world - cameraEye;
        const vec4 h   = scene->viewProjection() * vec4(float(rel(0)), float(rel(1)), float(rel(2)), 1);
        if (h(3) <= 0) return -1;
        const int x = int(std::floor((h(0) / h(3) * 0.5f + 0.5f) * float(img.w)));
        const int y = int(std::floor((h(1) / h(3) * 0.5f + 0.5f) * float(img.h)));
        if (x < 0 || y < 0 || x >= img.w || y >= img.h) return -1;
        return float(img(y, x)(0)) / 255.0f;
    }
    void Camera(const Vec3& eye, const Vec3& forward)
    {
        cameraEye = eye;
        SetCamera(0, eye, lookRotation(forward), 1.0);
    }
    Vec3 cameraEye = Vec3::Zero();
};
}  // namespace

// A pole on a plane: under its shadow the factor is below 0.1, outside above 0.9, the contact region near the pole is
// shadowed (no peter-panning beyond 2 texels of cascade 0)
TEST_F(Shadows, BoxOnPlane)
{
    Box(301, vec3(60, 0.5f, 60));
    Box(302, vec3(0.5f, 4, 0.5f));
    AddSet({instanceAt(vec3(0, -0.5f, 0), 301), instanceAt(vec3(0, 4, 0), 302)});
    Sun(Vec3(0.5, 1, 0));  // the shadow falls towards -x, 4 m beyond the pole
    // From the shadow's side (the pole hides the plane behind it, not the contact region)
    Camera(Vec3(-7, 7, 0.01), Vec3(6.5, -7, 0));
    const TemplatedImage<ucvec4> img = Capture(3);
    const LightingStatistics& st     = scene->lightingStatistics();
    ASSERT_TRUE(st.shadows);
    const double texel = st.cascades[0].texelSize;
    EXPECT_GT(texel, 0.0);
    EXPECT_LT(FactorAt(img, Vec3(-2.5, 0, 0)), 0.1f);
    EXPECT_LT(FactorAt(img, Vec3(-4.0, 0, 0.2)), 0.1f);
    EXPECT_GT(FactorAt(img, Vec3(-2.5, 0, 2)), 0.9f);
    EXPECT_GT(FactorAt(img, Vec3(-3, 0, -2)), 0.9f);
    EXPECT_GT(FactorAt(img, Vec3(-5, 0, 0)), 0.9f);
    EXPECT_GT(FactorAt(img, Vec3(3, 0, 3)), 0.9f);
    // Contact: 2 texels (+ the PCF's 1.5) beside the pole's shadowed face
    EXPECT_LT(FactorAt(img, Vec3(-0.5 - 3.5 * texel, 0, 0)), 0.5f) << "texel " << texel;
    // The pole's face towards the camera is in its own shadow (its back to the light)
    EXPECT_LT(FactorAt(img, Vec3(-0.5, 4, 0.1)), 0.1f);
}

// A caster nearer to the light than the cascades' near end (above the receivers' depth range, below the cap) is
// pancaked onto depth 0 and still shadows
TEST_F(Shadows, PancakedCasterBehindNear)
{
    LightingParameters p = scene->lightingParameters();
    p.shadowDistance     = 30;
    scene->setLightingParameters(p);
    Box(303, vec3(40, 0.5f, 40));
    Box(304, vec3(2, 2, 2));
    // 110 m above the plane (the depth range is about 2 x 30 m around the camera's frustum)
    AddSet({instanceAt(vec3(0, -0.5f, 0), 303), instanceAt(vec3(0, 110, 0), 304)});
    Sun(Vec3(0, 1, 0));
    Camera(Vec3(0.01, 12, 0), Vec3(0, -1, 0));
    const TemplatedImage<ucvec4> img = Capture(3);
    const LightingStatistics& st     = scene->lightingStatistics();
    ASSERT_TRUE(st.shadows);
    EXPECT_LT(st.cascades[0].center(1) + st.cascades[0].depthRange * 0.5, 110.0) << "the caster is above the range";
    EXPECT_LT(FactorAt(img, Vec3(0.3, 0, 0.3)), 0.1f);
    EXPECT_GT(FactorAt(img, Vec3(5, 0, 5)), 0.9f);
}

// The shadow pipeline table: a class without an entry casts nothing; registering it (S27's impostor, S34's skinned
// class) makes it cast. The pipeline classes are fixed (PipelineClass), so the test removes the opaque class's entry
// and registers it again
TEST_F(Shadows, PipelineTableSlotForNewClass)
{
    Box(305, vec3(60, 0.5f, 60));
    Box(306, vec3(1, 3, 1));
    AddSet({instanceAt(vec3(0, -0.5f, 0), 305), instanceAt(vec3(0, 3, 0), 306)});
    Sun(Vec3(0.6, 1, 0));
    Camera(Vec3(0.01, 25, 0), Vec3(0, -1, 0));
    const Vec3 shadowed(-3, 0, 0);
    EXPECT_LT(FactorAt(Capture(3), shadowed), 0.1f);
    scene->shadowPipelines().setEntry(0, std::nullopt);
    EXPECT_FALSE(scene->shadowPipelines().hasEntry(0));
    EXPECT_GT(FactorAt(Capture(3), shadowed), 0.9f) << "no entry: no shadow";
    ShadowPipelineTable::Entry e;
    e.vertexShader = "vulkan/scene/lighting/shadow_depth.vert";
    scene->shadowPipelines().setEntry(0, e);
    EXPECT_LT(FactorAt(Capture(3), shadowed), 0.1f) << "the registered entry casts";
}

// At 5 degrees a 20 m pole (4 m wide) 200 m towards the sun shadows the receivers on both sides of the cascade 0/1
// split near the camera (its caster volume reaches maxCasterHeight / sin(5 degrees) towards the light; review 1,
// finding 5)
TEST_F(Shadows, LongShadowCrossesCascadeSplit)
{
    Box(307, vec3(400, 0.5f, 400));
    Box(308, vec3(2, 10, 2));
    const double el = 5 * kPi / 180;
    AddSet({instanceAt(vec3(0, -0.5f, 0), 307), instanceAt(vec3(200, 10, 0), 308)});
    Sun(Vec3(std::cos(el), std::sin(el), 0));
    Camera(Vec3(0, 6, 0.01), Vec3(1, -0.4, 0));
    const TemplatedImage<ucvec4> img = Capture(3);
    const LightingStatistics& st     = scene->lightingStatistics();
    ASSERT_TRUE(st.shadows);
    const double split = st.cascades[0].splitFar;
    ASSERT_GT(split, 6.0);
    ASSERT_LT(split, 40.0);
    for (double x : {split * 0.6, split * 0.9, split * 1.1, split * 1.6})
        EXPECT_LT(FactorAt(img, Vec3(x, 0, 0)), 0.2f) << "x " << x << " (split " << split << ")";
    for (double x : {split * 0.9, split * 1.6}) EXPECT_GT(FactorAt(img, Vec3(x, 0, 3)), 0.9f) << "lit beside, x " << x;
}

// Review 2, finding 3: a 20 m pole inside cascade 0, 30 m from the camera, at 5 degrees: its shadow on the ground 100 m
// to 200 m away (beyond cascade 0's frustum slice; the receivers there sample cascades 1-3, which contain the pole: the
// de-duplication rejects only casters whose whole footprint lies inside cascade 0) is continuous
TEST_F(Shadows, LongShadowLeavesNearCascade)
{
    LightingParameters p = scene->lightingParameters();
    p.shadowDistance     = 1500;
    scene->setLightingParameters(p);
    Box(309, vec3(600, 0.5f, 600));
    Box(310, vec3(2, 10, 2));
    const double el = 5 * kPi / 180;
    AddSet({instanceAt(vec3(0, -0.5f, 0), 309), instanceAt(vec3(-30, 10, 0), 310)});
    Sun(Vec3(std::cos(el), std::sin(el), 0));
    // Beside the shadow line (the pole hides the ground behind it from a camera on the line)
    Camera(Vec3(0, 12, 14), Vec3(-1, -0.09, -0.07));
    const TemplatedImage<ucvec4> img = Capture(3);
    ASSERT_TRUE(scene->lightingStatistics().shadows);
    EXPECT_LT(scene->lightingStatistics().cascades[0].splitFar, 100.0);
    for (double x = -40; x >= -200; x -= 20) EXPECT_LT(FactorAt(img, Vec3(x, 0, 0)), 0.2f) << "x " << x;
    EXPECT_GT(FactorAt(img, Vec3(-120, 0, -12)), 0.9f) << "lit beside the shadow";
}

namespace
{
// The ridge of mountainRidgeHook (crest at x = 2048) as a continuous height (m)
double ridgeHeight(double x)
{
    const double d = std::abs(x - 2048);
    if (d >= 400) return 20;
    const double t = (160000 - d * d) / 400;
    return 20 + 300 * t * t / 160000;
}
// The far edge of the ridge's cast shadow towards -x for the light at elevation el from +x: going down from the crest,
// the first lit x after the shadow began (a dense march over the continuous profile)
double ridgeShadowEdge(double el)
{
    const double slope = std::tan(el);
    bool inShadow      = false;
    for (double x = 2047; x > 1000; x -= 0.05)
    {
        double horizon = -1e9;
        for (double t = 0.5; t < 1200; t += 0.5) horizon = std::max(horizon, ridgeHeight(x + t) - t * slope);
        const bool lit = ridgeHeight(x) >= horizon;
        if (!lit) inShadow = true;
        if (lit && inShadow) return x;
    }
    return 1000;
}

class ShadowsTerrain : public Shadows
{
   protected:
    void Ridge(const Vec3& camera)
    {
        const TerrainLayout layout = TerrainTestHooks::testLayout(16, 1);
        TerrainTestHooks::fillTestTerrain(*scene, layout,
                                          TerrainHeightSource::fromHook(TerrainTestHooks::mountainRidgeHook, 1),
                                          std::nullopt, camera, false);
    }
};
}  // namespace

// The terrain casts into the cascades (A5, finding 4: mountains taller than the instances' cap): the ridge's lee side
// at 30 degrees is in its shadow with terrainCastsInCascades, lit without (no horizon map: the cascades alone)
TEST_F(ShadowsTerrain, TerrainCastsInCascades)
{
    const double el   = 30 * kPi / 180;
    const double edge = ridgeShadowEdge(el);
    const Vec3 eye(edge - 40, ridgeHeight(edge - 40) + 50, 2048.5);
    Ridge(eye);
    Sun(Vec3(std::cos(el), std::sin(el), 0));
    Camera(eye, Vec3(1, -1.2, 0));
    const Vec3 shadowed(edge + 15, ridgeHeight(edge + 15), 2048);
    const Vec3 lit(edge - 15, ridgeHeight(edge - 15), 2048);
    TemplatedImage<ucvec4> img = Capture(3);
    EXPECT_LT(FactorAt(img, shadowed), 0.2f) << "edge " << edge;
    EXPECT_GT(FactorAt(img, lit), 0.9f);
    EXPECT_GT(scene->lightingStatistics().cascades[0].terrainNodes, 0u);
    LightingParameters p     = scene->lightingParameters();
    p.terrainCastsInCascades = false;
    scene->setLightingParameters(p);
    img = Capture(3);
    EXPECT_GT(FactorAt(img, shadowed), 0.9f) << "without terrain casters";
}

// The cliff near the camera (finding 4): the ridge's shadow edge on its lee side at 45 degrees lies within 2 cascade
// texels (plus the terrain's 1 m spacing and a pixel) of the dense CPU march
TEST_F(ShadowsTerrain, CliffNearCameraUsesCascade)
{
    const double el   = 45 * kPi / 180;
    const double edge = ridgeShadowEdge(el);
    const Vec3 eye(edge - 20, ridgeHeight(edge) + 30, 2048.5);
    Ridge(eye);
    Sun(Vec3(std::cos(el), std::sin(el), 0));
    Camera(eye, Vec3(0.8, -1, 0));
    const TemplatedImage<ucvec4> img = Capture(3);
    const LightingStatistics& st     = scene->lightingStatistics();
    ASSERT_TRUE(st.shadows);
    // The measured edge: the first x (towards +x, into the shadow) with a factor below 0.5
    double measured = 0;
    for (double x = edge - 10; x < edge + 10; x += 0.05)
        if (FactorAt(img, Vec3(x, ridgeHeight(x), 2048)) < 0.5f)
        {
            measured = x;
            break;
        }
    int c = 0;
    while (c < 3 && st.cascades[size_t(c)].splitFar < (Vec3(edge, ridgeHeight(edge), 2048) - eye).norm()) c++;
    const double tolerance = 2 * st.cascades[size_t(c)].texelSize + 1.0 + 0.3;
    EXPECT_NEAR(measured, edge, tolerance) << "cascade " << c << " texel " << st.cascades[size_t(c)].texelSize;
}

namespace
{
// The PrototypeFoliage class needs the foliage texture: its own scene
class ShadowsFoliage : public Shadows
{
};
}  // namespace

// The PrototypeFoliage class casts a perforated shadow (shadow_foliage.frag's alpha cutoff of the foliage texture)
TEST_F(ShadowsFoliage, FoliageAlphaShadow)
{
    scene->setFoliageTexture(7);
    Box(320, vec3(60, 0.5f, 60));
    // A horizontal leaf card (layer 0 of the foliage texture) 6 m above the plane and an opaque card (layer -1)
    MaterialData m;
    m.pipelineClass = uint8_t(PipelineClass::PrototypeFoliage);
    scene->addMaterial(321, "shadow_leaf", m);
    auto card = [](int layer, float half)
    {
        MeshLod lod;
        lod.endDistance       = 1e6f;
        const vec3 corners[4] = {vec3(-half, 0, -half), vec3(half, 0, -half), vec3(half, 0, half),
                                 vec3(-half, 0, half)};
        const vec2 uvs[4]     = {vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)};
        for (int i = 0; i < 4; ++i)
        {
            MeshVertex v;
            v.position = corners[i];
            v.normal   = packNormalOct(vec3(0, 1, 0));
            v.color    = packPrototypeFoliageColor(layer, 1, 0);
            v.uv       = uvs[i];
            lod.vertices.push_back(v);
        }
        for (uint32_t i : {0u, 2u, 1u, 0u, 3u, 2u}) lod.indices.push_back(i);
        MeshLods ml;
        ml.lods.push_back(lod);
        ml.sphereRadius = half * 1.5f;
        return ml;
    };
    scene->addMesh(321, "shadow_leaf_card", card(0, 3), 321);
    scene->addMesh(322, "shadow_opaque_card", card(-1, 3), 321);
    AddSet({instanceAt(vec3(0, -0.5f, 0), 320), instanceAt(vec3(-5, 6, 0), 321), instanceAt(vec3(5, 6, 0), 322)});
    // The light from the side: the shadows fall beside the cards (the camera sees them past the cards)
    Sun(Vec3(0.8, 1, 0));
    Camera(Vec3(0.01, 25, 0.5), Vec3(0, -1, 0));
    const TemplatedImage<ucvec4> side = Capture(3);
    auto mean                         = [&](double cx)
    {
        double sum = 0;
        int n      = 0;
        for (double x = -2; x <= 2; x += 0.25)
            for (double z = -2; z <= 2; z += 0.25)
            {
                const float f = FactorAt(side, Vec3(cx + x, 0, z));
                if (f < 0) continue;
                sum += f;
                n++;
            }
        return n ? sum / n : -1.0;
    };
    // The shadows fall 4.8 m towards -x of the cards
    const double leaf = mean(-5 - 4.8), opaque = mean(5 - 4.8);
    EXPECT_LT(opaque, 0.1) << "the opaque card";
    EXPECT_GT(leaf, 0.1) << "the leaf card is perforated";
    EXPECT_LT(leaf, 0.9) << "the leaf card casts";
}
