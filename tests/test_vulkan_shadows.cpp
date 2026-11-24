/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The sun and moon shadows of S36a2 on RenderScene (docs/stages/S36_lighting.plan.md, Tests): the multi-view culling
// of the shadow cascades against a CPU reference, the cascades' fitting (pure), the rendered cascade shadows (the
// ShadowFactor view read back) and the shadow pipeline table. A scene per test (RenderSceneTest), skipped without a
// Vulkan 1.3 device.

#include "saiga/vulkan/scene/CascadedShadows.h"
#include "saiga/vulkan/scene/HorizonShadows.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/ShadowPipelineTable.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

// The previous view's rectangle shrunk by the band as the de-duplication rectangle of v, its depth range up to the cap
// as the de-duplication interval
void dedupAgainst(ShadowView& v, const ShadowView& previous, double band)
{
    v.dedup     = true;
    v.dedupMin  = previous.rectMin + Vec2(band, band);
    v.dedupMax  = previous.rectMax - Vec2(band, band);
    v.dedupZMin = previous.zMin;
    v.dedupZMax = previous.zMax + previous.instanceCap;
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
            const double x = v.axisX.dot(c), y = v.axisY.dot(c), z = v.axisZ.dot(c);
            const double m = std::min({x - r - v.dedupMin(0), v.dedupMax(0) - x - r, y - r - v.dedupMin(1),
                                       v.dedupMax(1) - y - r, z - r - v.dedupZMin, v.dedupZMax - z - r});
            amb            = amb || std::abs(m) <= 1e-3 * std::max(d, 1.0);
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

class Cascades : public RenderSceneTest
{
   protected:
    ivec2 Size() const override { return ivec2(128, 128); }
    void SetUp() override
    {
        RenderSceneTest::SetUp();
        if (!scene || IsSkipped()) return;
        // The default lighting (not the fixture's golden mode): the staggered cascades are tested
        scene->setLightingParameters(LightingParameters());
    }
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

    LightingParameters off = scene->lightingParameters();
    off.sunShadows         = false;
    scene->setLightingParameters(off);
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
TEST(CascadesFit, SplitsMatchFormula)
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
TEST(CascadesFit, CoverCameraFrustum)
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
TEST(CascadesFit, StableUnderSubTexelMoves)
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
TEST(CascadesFit, IntervalKeepsMatrixCorrect)
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
// between its updates (its sphere around the camera has the margin interval * maxSpeed * dt). The cascade of frame 1
// is checked with frame 2's camera directly: in a fit of frame 2 it is rendered again, as cascade 2 turned away from
// the casters cascade 3 skipped (CascadesFit.SkippedCastersStayInPreviousCascade)
TEST(CascadesFit, StaleCascadeStillCoversFrustum)
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
    ASSERT_LE((next.camera - cs.cascades()[3].fitCamera).norm(), cs.cascades()[3].margin);
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

// Regression (the flight demo of S36a2): the shared depth range was the union of the cascades rendered in the frame, so
// with a static camera the caster volumes of cascades 0-2 shrank in every frame in which cascade 3 (interval 2) was not
// due, while cascade 3 kept skipping the casters inside cascade 2 (their sets alternated, shadows flickered). The range
// and the de-duplication depend only on the cascade layout: a static view gives the same cascades every frame
TEST(CascadesFit, StaticViewKeepsLayout)
{
    LightingParameters p;
    p.sunShadows = true;  // intervals {1, 1, 1, 2}
    CascadedShadows cs;
    std::array<CascadedShadows::Cascade, 4> first;
    int stale = 0;
    for (uint64_t f = 1; f <= 6; ++f)
    {
        cs.fit(cameraInput(Vec3(10, 40, -20), Vec3(0.2, -0.3, -1), Vec3(0.3, 0.5, 0.6), f), p, false);
        stale += cs.cascades()[3].updated ? 0 : 1;
        for (size_t c = 0; c < 4; ++c)
        {
            const CascadedShadows::Cascade& k = cs.cascades()[c];
            if (f == 1)
            {
                first[c] = k;
                continue;
            }
            EXPECT_EQ(k.zNear, first[c].zNear) << "cascade " << c << " frame " << f;
            EXPECT_EQ(k.zFar, first[c].zFar) << "cascade " << c << " frame " << f;
            EXPECT_EQ(k.casterTop, first[c].casterTop) << "cascade " << c << " frame " << f;
            EXPECT_EQ(k.cx, first[c].cx);
            EXPECT_EQ(k.cy, first[c].cy);
            EXPECT_EQ(k.dedup, first[c].dedup) << "cascade " << c << " frame " << f;
            EXPECT_EQ(k.dedupCx, first[c].dedupCx);
            EXPECT_EQ(k.dedupCy, first[c].dedupCy);
            EXPECT_EQ(k.dedupHalf, first[c].dedupHalf);
            EXPECT_EQ(k.dedupZ0, first[c].dedupZ0);
            EXPECT_EQ(k.dedupZ1, first[c].dedupZ1);
        }
    }
    EXPECT_EQ(stale, 3) << "cascade 3 keeps its interval";
    for (size_t c = 1; c < 4; ++c) EXPECT_TRUE(first[c].dedup) << c;
    // The shared range holds cascade 3's sphere also in the frames it is not rendered in
    EXPECT_GE(first[0].zNear, first[3].zNear);
    EXPECT_LE(first[0].zFar, first[3].zFar);
}

// Regression (S36a2 review, finding 13): the light basis followed every small move of the light, so with a moving sun
// (the flight demo's day in 10 minutes: about 0.01 degrees per frame) the texel grid rotated in every frame and every
// shadow edge crawled. The cascades' light moves in steps of kLightStepDeg: over 12 frames at 0.01 degrees per frame
// (static camera) the basis changes at most twice and stays within a step of the light; golden mode follows the light
TEST(CascadesFit, SlowLightMovesInSteps)
{
    LightingParameters p;
    p.sunShadows = true;
    CascadedShadows cs;
    Vec3 axis    = Vec3::Zero();
    int changes  = 0;
    auto lightAt = [](int f)
    {
        const double el = (20 + 0.01 * f) * kPi / 180;
        return Vec3(std::cos(el), std::sin(el), 0.3);
    };
    for (int f = 1; f <= 12; ++f)
    {
        const Vec3 light = lightAt(f).normalized();
        cs.fit(cameraInput(Vec3(0, 30, 0), Vec3(0.2, -0.3, -1), light, uint64_t(f)), p, false);
        const Vec3 x = cs.cascades()[0].axisX;
        if (f > 1 && (x - axis).norm() > 0) changes++;
        axis = x;
        EXPECT_LE(std::acos(std::min(1.0, cs.fittedLight().dot(light))) * 180 / kPi,
                  CascadedShadows::kLightStepDeg + 1e-9)
            << f;
    }
    EXPECT_GE(changes, 1);
    EXPECT_LE(changes, 2);
    p.forceFullUpdates = true;
    cs.fit(cameraInput(Vec3(0, 30, 0), Vec3(0.2, -0.3, -1), lightAt(13), 13), p, false);
    EXPECT_LT((cs.fittedLight() - lightAt(13).normalized()).norm(), 1e-12) << "golden mode: the exact light";
}

// Regression (S36a2 review, finding 12): the band and the edge margin of the receivers (fractions of the half extent):
// with cascadeBlend 0 the cascade's weight reached the rectangle's edge, where the PCF footprint and the normal offset
// read beyond the layer (clamp to edge: streaks); now the band is at least the margin plus 4 texels
TEST(CascadesFit, BlendBandKeepsTheFootprintInside)
{
    LightingParameters p;
    p.cascadeBlend = 0;
    for (int res : {256, 2048})
    {
        const double texel = 2.0 / res;  // one texel, a fraction of the half extent
        EXPECT_NEAR(CascadedShadows::edgeMargin(p, res), 3 * texel, 1e-12);
        EXPECT_NEAR(CascadedShadows::blendBand(p, res), 7 * texel, 1e-12);
    }
    // CodeRabbit #85 round 2: the margin holds the whole normal offset (20 texels moved the taps 17 texels past the
    // fixed 3-texel margin), and the band stays wider than it
    p.normalOffsetTexels = 20;
    EXPECT_NEAR(CascadedShadows::edgeMargin(p, 2048), 22 * 2.0 / 2048, 1e-12);
    EXPECT_NEAR(CascadedShadows::blendBand(p, 2048), 26 * 2.0 / 2048, 1e-12);
    p.normalOffsetTexels = 1;
    p.cascadeBlend = 0.1f;
    EXPECT_NEAR(CascadedShadows::blendBand(p, 2048), 0.1, 1e-7);
    // Finding 10: the fit's resolution is the image's
    p.cascadeResolution = 128;
    EXPECT_EQ(CascadedShadows::clampedResolution(p), 256);
    p.cascadeResolution = 8192;
    EXPECT_EQ(CascadedShadows::clampedResolution(p), 4096);
}

// A caster skipped by cascade c lies in cascade c - 1's caster volume in every frame in which c's map is sampled: with
// a moving and turning camera and a moving light, each cascade's stored de-duplication volume (rectangle times light
// depth interval) is inside cascade c - 1's current rectangle (shrunk by the blend band) times its depth range up to
// the instance cap (sampled points); a staggered cascade whose skipped casters would leave c - 1 is rendered again,
// otherwise it keeps its interval
TEST(CascadesFit, SkippedCastersStayInPreviousCascade)
{
    LightingParameters p;
    p.sunShadows = true;
    CascadedShadows cs;
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(-1, 1);
    int stale = 0, checked = 0, violations = 0;
    for (uint64_t f = 1; f <= 60; ++f)
    {
        // Frames 1-20 straight ahead, 21-40 turning by 2 degrees per frame, 41-60 straight again; the sun moves
        const double t       = double(f);
        const double heading = f <= 20 ? 0.0 : (f <= 40 ? (t - 20) * 2 * kPi / 180 : 40 * kPi / 180);
        const Vec3 forward(std::sin(heading), -0.2, -std::cos(heading));
        const Vec3 eye   = Vec3(1.5 * t, 50, -1.5 * t);
        const Vec3 light = Vec3(0.3 + 0.0005 * t, 0.5, 0.6);
        CascadeInput in  = cameraInput(eye, forward, light, f);
        in.maxSpeed      = 130;
        cs.fit(in, p, false);
        stale += cs.cascades()[3].updated ? 0 : 1;
        const double blend = p.cascadeBlend;
        for (size_t c = 1; c < 4; ++c)
        {
            const CascadedShadows::Cascade &k = cs.cascades()[c], &prev = cs.cascades()[c - 1];
            if (!k.dedup) continue;
            const double prevHalf = prev.halfExtent * (1 - blend);
            for (int i = 0; i < 64; ++i)
            {
                const double a = i < 8 ? ((i & 1) ? 1 : -1) : u(rng), b = i < 8 ? ((i & 2) ? 1 : -1) : u(rng);
                const double w = i < 8 ? ((i & 4) ? k.dedupZ1 : k.dedupZ0)
                                       : k.dedupZ0 + (k.dedupZ1 - k.dedupZ0) * (0.5 + 0.5 * u(rng));
                const Vec3 q =
                    k.axisX * (k.dedupCx + a * k.dedupHalf) + k.axisY * (k.dedupCy + b * k.dedupHalf) + k.axisZ * w;
                const double tol  = 1e-5;
                const bool inside = std::abs(prev.axisX.dot(q) - prev.cx) <= prevHalf + tol &&
                                    std::abs(prev.axisY.dot(q) - prev.cy) <= prevHalf + tol &&
                                    prev.axisZ.dot(q) >= prev.zFar - tol && prev.axisZ.dot(q) <= prev.casterTop + tol;
                violations += inside ? 0 : 1;
                checked++;
            }
        }
    }
    EXPECT_EQ(violations, 0);
    EXPECT_GT(checked, 60 * 64);
    EXPECT_GT(stale, 10) << "the staggered cascade keeps its interval while the camera flies straight";
}

// The light basis near the zenith: orthonormal, and a light tilted by 0.01 rad changes the basis by about as much (no
// flip); towards the north horizon the fallback keeps it finite
TEST(CascadesFit, BasisStableNearZenith)
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
TEST(CascadesFit, ShadowLightSwitchHasHysteresis)
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
class Shadows : public RenderSceneTest
{
   protected:
    ivec2 Size() const override { return ivec2(256, 256); }
    int Samples() const override { return 1; }

    void SetUp() override
    {
        RenderSceneTest::SetUp();
        if (!scene || IsSkipped()) return;
        LightingParameters p;
        p.sunShadows        = true;
        p.cascadeResolution = 1024;
        p.shadowDistance    = 200;
        scene->setLightingParameters(p);
        scene->setLightingDebug({LightingDebugView::ShadowFactor, 0});
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
    // The factor (0..1) at a world point of the last frame (its matrices), NaN outside the image: every comparison
    // with it fails (a point a check needs must be in view), scans skip it with std::isnan
    float FactorAt(const TemplatedImage<ucvec4>& img, const Vec3& world)
    {
        const float none = std::numeric_limits<float>::quiet_NaN();
        const Vec3 rel   = world - cameraEye;
        const vec4 h     = scene->viewProjection() * vec4(float(rel(0)), float(rel(1)), float(rel(2)), 1);
        if (h(3) <= 0) return none;
        const int x = int(std::floor((h(0) / h(3) * 0.5f + 0.5f) * float(img.w)));
        const int y = int(std::floor((h(1) / h(3) * 0.5f + 0.5f) * float(img.h)));
        if (x < 0 || y < 0 || x >= img.w || y >= img.h) return none;
        return float(img(y, x)(0)) / 255.0f;
    }
    void Camera(const Vec3& eye, const Vec3& forward) { SetCamera(0, eye, lookRotation(forward), 1.0); }
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

// Regression (S36a2 review, finding 9): the staggered cascade's margin (interval x speed x dt) took the frame time and
// the camera's speed from the wall clock, so its radius and texel depended on the machine's frame time (lavapipe's
// 0.1 s frames gave a 10x margin). With RenderScene::setFrameTime (the app's frame dt) the fit uses the frames' clock:
// dt is the given step and the speed of a static camera is the minimum
TEST_F(Shadows, StaggeredMarginUsesTheFrameTime)
{
    LightingParameters p = scene->lightingParameters();
    p.forceFullUpdates   = false;
    scene->setLightingParameters(p);
    Box(313, vec3(60, 0.5f, 60));
    AddSet({instanceAt(vec3(0, -0.5f, 0), 313)});
    Sun(Vec3(0.5, 1, 0.2));
    Camera(Vec3(-7, 7, 0.01), Vec3(6.5, -7, 0));
    for (int i = 0; i < 6; ++i)
    {
        scene->setFrameTime(1.0 / 60);
        RenderFrame();
    }
    scene->waitIdle();
    const CascadeInput& in = scene->cascadedShadows().fittedCamera();
    EXPECT_DOUBLE_EQ(in.dt, 1.0 / 60);
    EXPECT_DOUBLE_EQ(in.maxSpeed, 50);
}

// Regression (CodeRabbit #85): a changed projection (here the field of view) kept the staggered cascades on the slices
// of the old frustum until they were due; every cascade is refitted in the frame of the change
TEST_F(Shadows, ProjectionChangeRefitsEveryCascade)
{
    LightingParameters p = scene->lightingParameters();
    p.forceFullUpdates   = false;
    p.cascadeInterval    = {1, 8, 8, 8};
    scene->setLightingParameters(p);
    Box(314, vec3(60, 0.5f, 60));
    AddSet({instanceAt(vec3(0, -0.5f, 0), 314)});
    Sun(Vec3(0.5, 1, 0.2));
    const Vec3 forward(6.5, -7, 0);
    Camera(Vec3(-7, 7, 0.01), forward);
    for (int i = 0; i < 3; ++i)
    {
        scene->setFrameTime(1.0 / 60);
        RenderFrame();
    }
    scene->waitIdle();
    const LightingStatistics before = scene->lightingStatistics();
    ASSERT_TRUE(before.shadows);
    ASSERT_EQ(before.cascadeCount, 4);
    // A wider field of view: every slice's bounding sphere grows (a narrower one refits the staggered cascades anyway:
    // their de-duplication against the shrunk cascade 0 no longer holds)
    SetCamera(0, cameraEye, lookRotation(forward), 1.4);
    scene->setFrameTime(1.0 / 60);
    RenderFrame();
    scene->waitIdle();
    const LightingStatistics& st = scene->lightingStatistics();
    for (size_t c = 0; c < 4; ++c)
        EXPECT_GT(st.cascades[c].halfExtent, before.cascades[c].halfExtent * 1.1f) << "cascade " << c;
}

// Regression (S36a2 review, findings 10 and 11): a C++ caller's resolution outside 256..4096 fitted the texel snap
// and the biases to another texel than the image's; and while frozen a resolution or count change (the images are
// recreated, every cascade invalid) rendered no cascade until unfrozen. Frozen, then 128 texels (an image of 256): the
// cascades are refitted once for the frozen camera with the image's texel, and the pole's shadow stays
TEST_F(Shadows, FrozenCascadesFollowAResolutionChange)
{
    Box(311, vec3(60, 0.5f, 60));
    Box(312, vec3(0.5f, 4, 0.5f));
    AddSet({instanceAt(vec3(0, -0.5f, 0), 311), instanceAt(vec3(0, 4, 0), 312)});
    Sun(Vec3(0.5, 1, 0));
    Camera(Vec3(-7, 7, 0.01), Vec3(6.5, -7, 0));
    const Vec3 shadow(-2, 0, 0);
    EXPECT_LT(FactorAt(Capture(2), shadow), 0.2f);
    scene->setFreezeCascades(true);
    LightingParameters p = scene->lightingParameters();
    p.cascadeResolution  = 128;
    scene->setLightingParameters(p);
    const TemplatedImage<ucvec4> img = Capture(2);
    const LightingStatistics& st     = scene->lightingStatistics();
    ASSERT_TRUE(st.shadows);
    ASSERT_TRUE(st.cascades[0].valid);
    EXPECT_NEAR(st.cascades[0].texelSize * 256, 2 * st.cascades[0].halfExtent, 1e-6) << "the image's texel";
    EXPECT_LT(FactorAt(img, shadow), 0.2f) << "frozen cascades after the change";
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

// Regression (the flight demo of S36a2): with a static camera and the default intervals (cascade 3 every second frame)
// the caster sets of cascades 0-2 alternated with cascade 3's updates and the shadows flickered. Now every frame draws
// the same instances into cascades 0-2 (cascade 3 the same in each of its frames) and the shadow image is the same
TEST_F(Shadows, StaticViewSameCastersEveryFrame)
{
    Box(311, vec3(300, 0.5f, 300));
    Box(312, vec3(1, 6, 1));
    std::vector<StaticInstance> set = {instanceAt(vec3(0, -0.5f, 0), 311)};
    for (int i = 0; i < 12; ++i)
        for (int j = -3; j <= 3; ++j) set.push_back(instanceAt(vec3(float(j) * 9, 6, -6.0f - 16.0f * float(i)), 312));
    AddSet(set);
    Sun(Vec3(0.4, 0.5, -0.6));
    Camera(Vec3(0.01, 14, 6), Vec3(0, -0.35, -1));
    Capture(3);
    TemplatedImage<ucvec4> reference;
    std::array<uint32_t, 4> counts{}, rendered{};
    int frames3 = 0;
    for (int f = 0; f < 4; ++f)
    {
        const TemplatedImage<ucvec4> img = Capture(1);
        const RenderStatistics st        = scene->statistics();
        ASSERT_GT(img.w, 0);
        ASSERT_GT(img.h, 0);
        if (f == 0)
        {
            reference = img;
            for (size_t c = 0; c < 3; ++c) counts[c] = st.cascadeInstances[c];
        }
        else
        {
            for (size_t c = 0; c < 3; ++c)
                EXPECT_EQ(st.cascadeInstances[c], counts[c]) << "cascade " << c << " frame " << f;
            bool same = img.h == reference.h && img.w == reference.w;
            for (int y = 0; same && y < img.h; ++y)
                for (int x = 0; same && x < img.w; ++x) same = img(y, x) == reference(y, x);
            EXPECT_TRUE(same) << "frame " << f;
        }
        if (st.cascadeInstances[3] > 0)
        {
            if (frames3++ == 0) rendered[3] = st.cascadeInstances[3];
            EXPECT_EQ(st.cascadeInstances[3], rendered[3]) << "frame " << f;
        }
    }
    for (size_t c = 0; c < 3; ++c) EXPECT_GT(counts[c], 0u) << c;
    EXPECT_EQ(frames3, 2) << "cascade 3 every second frame";
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

// Regression (S36a2 review, major 5): in the blend band of the last cascade the cascade faded towards lit while the
// receivers still used the horizon map's G (from cascadeCasterDistance on), so the ridge's shadow (300-650 m away,
// in neither G nor the faded cascade) had a lit gap along the border; outside the rectangles R shadowed it again. A
// line of ground points in the ridge's shadow at 10 degrees (x = 1398, along z away from the camera: across every
// cascade, their bands and beyond) is shadowed everywhere; with the terrain's de-duplication (finding 8)
TEST_F(ShadowsTerrain, NoLitGapAtTheLastCascadeBorder)
{
    const double el = 10 * kPi / 180;
    const double x0 = 1398;
    // x0 is deep in the ridge's shadow, cast by the slope 250-650 m towards the light (beyond R's 64 m, before G's
    // 1371 m)
    double horizon = -1e9;
    for (double t = 0.5; t < 1200; t += 0.5) horizon = std::max(horizon, ridgeHeight(x0 + t) - t * std::tan(el));
    ASSERT_GT(horizon, ridgeHeight(x0) + 20);
    const Vec3 eye(x0, 100, 2048.5);
    Ridge(eye);
    LightingParameters p = scene->lightingParameters();
    p.shadowDistance     = 300;
    p.cascadeResolution  = 512;
    p.horizonResolution  = 256;
    p.horizonExtent      = 8192;
    p.forceFullUpdates   = true;
    scene->setLightingParameters(p);
    Sun(Vec3(std::cos(el), std::sin(el), 0));
    SetCamera(0, eye, lookRotation(Vec3(0, -0.45, 1)), 1.2);
    // (two frames: the horizon map is sampled from the frame after its computation)
    const TemplatedImage<ucvec4> img = Capture(2);
    const LightingStatistics& st     = scene->lightingStatistics();
    ASSERT_TRUE(st.shadows);
    int checked = 0, inBand = 0;
    for (double z = 2048 + 60; z < 2048 + 1200; z += 2)
    {
        const Vec3 q(x0, ridgeHeight(x0), z);
        const float f = FactorAt(img, q);
        if (std::isnan(f)) continue;
        checked++;
        // Beyond the last cascade's band start (its half extent along z times 1 - blend; the rectangle is centred near
        // the camera)
        const double half = st.cascades[size_t(st.cascadeCount - 1)].halfExtent;
        if (std::abs(z - eye(2)) > half * (1 - p.cascadeBlend) - 40 && std::abs(z - eye(2)) < half + 40) inBand++;
        EXPECT_LT(f, 0.25f) << "z " << z;
    }
    EXPECT_GT(checked, 100);
    EXPECT_GT(inBand, 5) << "the line crosses the last cascade's band";
    // The terrain nodes inside the previous cascade are not drawn again (finding 8; the line above is shadowed through
    // every cascade, so no caster was lost)
    uint32_t deduplicated = 0;
    for (int c = 1; c < st.cascadeCount; ++c) deduplicated += st.cascades[size_t(c)].terrainDeduplicated;
    EXPECT_GT(deduplicated, 0u);
    EXPECT_EQ(st.cascades[0].terrainDeduplicated, 0u);
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
                if (std::isnan(f)) continue;
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

//==================================================================================================
// The far terrain horizon map (A6)

namespace
{
// A 10% slope falling towards +x (facing a sun from +x), 200 m high at x = 0
int32_t slopeTowardsSunHook(uint64_t, int64_t x, int64_t)
{
    return int32_t((200 * 1024) - x * 1024 / 10);
}

class HorizonTest : public Shadows
{
   protected:
    void SetUp() override
    {
        Shadows::SetUp();
        if (!scene) return;
        LightingParameters p = scene->lightingParameters();
        p.sunShadows         = false;  // the horizon map alone
        p.horizonShadows     = true;
        p.horizonResolution  = 256;
        p.horizonExtent      = 8192;  // 32 m texels
        p.forceFullUpdates   = true;
        scene->setLightingParameters(p);
        scene->setLightingDebug({LightingDebugView::HorizonShadow, 0});
    }
    // A terrain of the hook (4 km, 2 m spacing): the pyramid (the horizon map's source) and the camera's tiles
    void Terrain(TerrainHeightHook hook, const Vec3& camera)
    {
        layout  = TerrainTestHooks::testLayout(16, 2);
        pyramid = TerrainTestHooks::testPyramid(hook, layout);
        TerrainTestHooks::fillTestTerrain(*scene, layout, TerrainHeightSource::fromHook(hook, 1), std::nullopt, camera,
                                          false);
    }
    // The light from +x at the elevation (degrees)
    void Elevation(double deg) { Sun(Vec3(std::cos(deg * kPi / 180), std::sin(deg * kPi / 180), 0)); }
    // The front map (R, G) and its transform
    std::vector<vec2> Front(HorizonShadows::MapTransform& t)
    {
        HorizonShadows& h = scene->horizonShadows();
        t                 = h.transform(h.frontLayer());
        return h.readLayerForTests(h.frontLayer());
    }
    // The dense CPU reference of a texel centre (x, z): max over t >= t0 (1 m steps) of the pyramid surface minus t
    // tan(el), relative to the map's reference
    double Reference(double x, double z, double el, double t0, const HorizonShadows::MapTransform& t) const
    {
        double best = -1e9;
        for (double s = t0; s < 6000; s += 1.0)
        {
            const double px = x + s;
            if (px > layout.sizeChunks * 256.0) break;
            best = std::max(best, TerrainTestHooks::pyramidSurfaceHeight(layout, pyramid, px, z) - s * std::tan(el));
        }
        return best - double(t.refUnits) * kHeightUnit;
    }
    double Surface(double x, double z) const { return TerrainTestHooks::pyramidSurfaceHeight(layout, pyramid, x, z); }
    TerrainLayout layout;
    TerrainPyramidData pyramid;
};
}  // namespace

// The conservative march against a dense CPU march over the pyramid's surface (finding 5): for 64 texels across the
// ridge the GPU's shadow heights (R and G) are never below the reference (from 64 m, from cascadeCasterDistance) by
// more than 0.5 m (no crest missed) and above the reference that starts one footprint earlier by at most that
// footprint (the 2 x 2 cells of a step reach about 90 m behind it beyond 1 km: 90 m tan(el)) plus 2 m (the cells
// hold the pyramid's samples, the reference its interpolated surface)
TEST_F(HorizonTest, MatchCpuReference)
{
    const Vec3 eye(1500, 450, 2048);
    Terrain(TerrainTestHooks::mountainRidgeHook, eye);
    for (double deg : {3.0, 12.0})
    {
        Elevation(deg);
        Camera(eye, Vec3(1, -0.5, 0));
        Capture(2);
        HorizonShadows::MapTransform t;
        const std::vector<vec2> map = Front(t);
        ASSERT_TRUE(t.valid);
        ASSERT_EQ(map.size(), size_t(t.resolution * t.resolution));
        const double el = deg * kPi / 180, footprint = 1.4142 * 64, tolerance = 2.0 + footprint * std::tan(el);
        int checked = 0;
        for (int k = 0; k < 64; ++k)
        {
            const int i = int((1000.0 + k * 16.0 - t.cornerX) / t.texel), j = int((2048.0 - t.cornerZ) / t.texel);
            if (i < 0 || j < 0 || i >= t.resolution || j >= t.resolution) continue;
            const double x = t.cornerX + (i + 0.5) * t.texel, z = t.cornerZ + (j + 0.5) * t.texel;
            const vec2 gpu = map[size_t(j) * size_t(t.resolution) + size_t(i)];
            const double r = Reference(x, z, el, 64, t), g = Reference(x, z, el, 1371, t);
            const double rUp = Reference(x, z, el, 64 - 45, t), gUp = Reference(x, z, el, 1371 - 32 - 45, t);
            EXPECT_GE(gpu(0), r - 0.5) << deg << " deg, x " << x;
            EXPECT_LE(gpu(0), rUp + tolerance) << deg << " deg, x " << x;
            if (g > -1e8)
            {
                EXPECT_GE(gpu(1), g - 0.5) << deg << " deg, x " << x;
                EXPECT_LE(gpu(1), gUp + tolerance) << deg << " deg, x " << x;
            }
            checked++;
        }
        EXPECT_GT(checked, 50);
    }
}

// The far shadow edge (the first lit texel going away from the ridge) moves by less than 2 texels when the light
// turns by 0.1 degrees (the conservative march does not flicker with the light direction)
TEST_F(HorizonTest, FarEdgeStableUnderSmallRotation)
{
    const Vec3 eye(1500, 450, 2048);
    Terrain(TerrainTestHooks::mountainRidgeHook, eye);
    auto edge = [&](double deg, double azimuthDeg)
    {
        const double el = deg * kPi / 180, az = azimuthDeg * kPi / 180;
        Sun(Vec3(std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)));
        Camera(eye, Vec3(1, -0.5, 0));
        Capture(2);
        HorizonShadows::MapTransform t;
        const std::vector<vec2> map = Front(t);
        // (no map: a failure, not an out-of-range scan)
        const bool valid =
            t.valid && t.texel > 0 && t.resolution > 0 && map.size() == size_t(t.resolution) * size_t(t.resolution);
        EXPECT_TRUE(valid) << "no horizon map at " << deg << ", " << azimuthDeg;
        if (!valid) return -1.0;
        const int j = int((2048.0 - t.cornerZ) / t.texel);
        bool shadow = false;
        for (int i = int((2040.0 - t.cornerX) / t.texel); i > 0; --i)
        {
            const double x = t.cornerX + (i + 0.5) * t.texel, z = t.cornerZ + (j + 0.5) * t.texel;
            const double y = Surface(x, z) - double(t.refUnits) * kHeightUnit;
            const bool lit = map[size_t(j) * size_t(t.resolution) + size_t(i)](0) < y;
            if (!lit) shadow = true;
            if (lit && shadow) return double(i);
        }
        return 0.0;
    };
    const double a = edge(4, 0), b = edge(4.1, 0), c = edge(4, 0.1);
    EXPECT_GT(a, 0);
    EXPECT_LE(std::abs(a - b), 2.0);
    EXPECT_LE(std::abs(a - c), 2.0);
}

// Flat ground and a 10% slope facing the sun are lit at 2, 5 and 10 degrees (finding 4: the one-sided band, the
// march from 64 m)
TEST_F(HorizonTest, FlatGroundIsLitAtLowSun)
{
    for (TerrainHeightHook hook :
         {TerrainHeightHook(TerrainTestHooks::flatHook), TerrainHeightHook(slopeTowardsSunHook)})
    {
        scene->clearTerrain();
        const Vec3 eye(2000, 0, 2000);
        Terrain(hook, eye);
        const Vec3 cam(2000, Surface(2000, 2000) + 60, 2000);
        for (double deg : {2.0, 5.0, 10.0})
        {
            Elevation(deg);
            Camera(cam, Vec3(0.3, -1, 0.2));
            const TemplatedImage<ucvec4> img = Capture(2);
            for (const Vec2& p : {Vec2(2010, 2010), Vec2(2030, 1990), Vec2(1990, 2025)})
                EXPECT_GT(FactorAt(img, Vec3(p(0), Surface(p(0), p(1)), p(1))), 0.98f) << deg << " deg";
        }
    }
}

// A valley point behind the ridge (towards -x) is lit at noon and in the ridge's shadow at 3 degrees (the long shadow
// at sunset)
TEST_F(HorizonTest, LongShadowAtSunset)
{
    const Vec3 point(1500, 0, 2048);
    const Vec3 eye(1450, 120, 2048.5);
    Terrain(TerrainTestHooks::mountainRidgeHook, eye);
    const Vec3 ground(point(0), Surface(point(0), point(2)), point(2));
    Elevation(80);
    Camera(eye, Vec3(1, -1.5, 0));
    EXPECT_GT(FactorAt(Capture(2), ground), 0.9f) << "noon";
    Elevation(3);
    EXPECT_LT(FactorAt(Capture(2), ground), 0.1f) << "sunset";
}

// The incremental update (rows per frame) gives the same map as a full update; its rows per frame follow the
// parameter
TEST_F(HorizonTest, IncrementalEqualsFull)
{
    const Vec3 eye(1500, 450, 2048);
    Terrain(TerrainTestHooks::mountainRidgeHook, eye);
    Elevation(6);
    Camera(eye, Vec3(1, -0.5, 0));
    Capture(2);
    HorizonShadows::MapTransform t;
    const std::vector<vec2> full = Front(t);
    // The same light again after another one, incrementally (32 rows per frame: 8 frames)
    LightingParameters p  = scene->lightingParameters();
    p.forceFullUpdates    = false;
    p.horizonRowsPerFrame = 32;
    scene->setLightingParameters(p);
    // (a jump of the light resets its angular speed: the base rows per frame apply)
    Elevation(20);
    RenderFrame();
    Elevation(6);
    int frames = 0;
    while (frames < 40)
    {
        RenderFrame();
        frames++;
        scene->waitIdle();
        const HorizonStatistics st = scene->lightingStatistics().horizon;
        // (the rows of each frame follow horizonRowsPerFrame: the steady light adds none)
        if (st.computing)
        {
            EXPECT_EQ(st.rowsThisFrame, 32) << frames;
        }
        const vec3 l(float(std::cos(6 * kPi / 180)), float(std::sin(6 * kPi / 180)), 0);
        if (!st.computing && (st.light - l).norm() < 1e-4f) break;
    }
    EXPECT_GE(frames, 8);
    EXPECT_LT(frames, 40);
    HorizonShadows::MapTransform u;
    const std::vector<vec2> incremental = Front(u);
    ASSERT_EQ(full.size(), incremental.size());
    float maxDiff = 0;
    for (size_t k = 0; k < full.size(); ++k)
        if (full[k](0) > -1e8f)
            maxDiff =
                std::max({maxDiff, std::abs(full[k](0) - incremental[k](0)), std::abs(full[k](1) - incremental[k](1))});
    EXPECT_LE(maxDiff, 1e-3f);
}

// After a light change the factor at a receiver moves monotonically from the previous map's value to the new map's,
// not before the new map was computed (16 rows per frame of 256: 16 frames); the blend of the shadow heights moves the
// shadow's edge continuously over the ground (at one point the factor changes within a few frames)
TEST_F(HorizonTest, BlendsDuringRecompute)
{
    const Vec3 point(1500, 0, 2048);
    const Vec3 eye(1450, 120, 2048.5);
    Terrain(TerrainTestHooks::mountainRidgeHook, eye);
    const Vec3 ground(point(0), Surface(point(0), point(2)), point(2));
    Elevation(80);
    Camera(eye, Vec3(1, -1.5, 0));
    const float lit = FactorAt(Capture(2), ground);
    EXPECT_GT(lit, 0.9f);
    LightingParameters p  = scene->lightingParameters();
    p.forceFullUpdates    = false;
    p.horizonRowsPerFrame = 16;
    scene->setLightingParameters(p);
    // (the jump to 3 degrees resets the light's angular speed: the base rows per frame apply)
    Elevation(3);
    const uint64_t completed = scene->lightingStatistics().horizon.completed;
    std::vector<float> f;
    int completion = -1;  // the sample of the frame that completed the new map
    for (int i = 0; i < 40; ++i)
    {
        f.push_back(FactorAt(Capture(1), ground));
        if (completion < 0 && scene->lightingStatistics().horizon.completed > completed) completion = i;
    }
    ASSERT_GE(completion, 0) << "the new map completed";
    // Up to the frame that completes the new map (it binds the maps of the previous update) the factor is the previous
    // map's
    for (int i = 0; i <= completion; ++i) EXPECT_GT(f[size_t(i)], 0.9f) << i << " (completed in " << completion << ")";
    EXPECT_LT(f.back(), 0.1f) << "the new map is shown in the end";
    size_t first = f.size();
    for (size_t i = 1; i < f.size(); ++i)
    {
        EXPECT_LE(f[i], f[i - 1] + 1.0f / 255) << i;
        if (f[i] < 0.5f && first == f.size()) first = i;
    }
    EXPECT_GE(first, 14u) << "the change follows the recompute";
}

// Regression (CodeRabbit #85): a changed cascadeCasterDistance (where G starts) did not recompute the horizon map, G
// kept the old distance
TEST_F(HorizonTest, CasterDistanceChangeRecomputes)
{
    const Vec3 eye(1450, 120, 2048.5);
    Terrain(TerrainTestHooks::mountainRidgeHook, eye);
    Elevation(3);
    Camera(eye, Vec3(1, -1.5, 0));
    Capture(2);
    const uint64_t before   = scene->lightingStatistics().horizon.recomputes;
    LightingParameters p    = scene->lightingParameters();
    p.cascadeCasterDistance = 800;
    scene->setLightingParameters(p);
    // (the statistics are written before the frame's update)
    Capture(2);
    EXPECT_GT(scene->lightingStatistics().horizon.recomputes, before);
    HorizonShadows::MapTransform t;
    Front(t);
    EXPECT_EQ(t.casterDistance, 800.0f);
}

// Regression (CodeRabbit #85): with the camera in another space (whose pyramid the atlas does not hold: nothing is
// updated) the lighting block still carried the previous space's horizon map
TEST_F(HorizonTest, NoMapInAnotherSpace)
{
    const Vec3 eye(1450, 120, 2048.5);
    Terrain(TerrainTestHooks::mountainRidgeHook, eye);
    Elevation(3);
    Camera(eye, Vec3(1, -1.5, 0));
    Capture(2);
    const LightingParameters p = scene->lightingParameters();
    SceneLighting block;
    scene->horizonShadows().fillBlock(block, eye, 0, p, true, true);
    EXPECT_GE(block.horizonFront(3), 0.0f) << "the map in its space";
    scene->horizonShadows().fillBlock(block, eye, 1, p, true, true);
    EXPECT_LT(block.horizonFront(3), 0.0f) << "no map in space 1";
}

// A small camera move recomputes nothing; leaving the inner half of the map moves the anchor and recomputes, and
// during that recompute a shadowed world point stays shadowed (each map keeps its own transform; finding 15)
TEST_F(HorizonTest, NoRecomputeForSmallCameraMovesAndAnchorMoveKeepsShadows)
{
    const Vec3 point(1500, 0, 2048);
    const Vec3 eye(1450, 120, 2048.5);
    Terrain(TerrainTestHooks::mountainRidgeHook, eye);
    const Vec3 ground(point(0), Surface(point(0), point(2)), point(2));
    Elevation(3);
    Camera(eye, Vec3(1, -1.5, 0));
    EXPECT_LT(FactorAt(Capture(2), ground), 0.1f);
    LightingParameters p  = scene->lightingParameters();
    p.forceFullUpdates    = false;
    p.horizonRowsPerFrame = 16;
    scene->setLightingParameters(p);
    const uint64_t before = scene->lightingStatistics().horizon.recomputes;
    const ivec2 corner    = scene->lightingStatistics().horizon.corner;
    // 100 m: inside the inner half (2 km)
    Camera(eye + Vec3(60, 0, 80), Vec3(1, -1.5, 0));
    Capture(3);
    EXPECT_EQ(scene->lightingStatistics().horizon.recomputes, before);
    // 2.5 km along z: a new anchor; the point stays shadowed through the recompute
    const Vec3 far(eye(0), eye(1), eye(2) - 2500);
    std::vector<float> f;
    Camera(eye + Vec3(0, 0, 0.5), Vec3(1, -1.5, 0));
    for (int i = 0; i < 24; ++i)
    {
        // The anchor follows the far camera position; the frames are viewed from near the point
        scene->horizonShadows();
        Camera(i == 0 ? far : eye + Vec3(0, 0, 0.5), Vec3(1, -1.5, 0));
        const TemplatedImage<ucvec4> img = Capture(1);
        if (i > 0) f.push_back(FactorAt(img, ground));
    }
    EXPECT_GT(scene->lightingStatistics().horizon.recomputes, before);
    EXPECT_NE(scene->lightingStatistics().horizon.corner, corner);
    for (float v : f) EXPECT_LT(v, 0.1f);
}

// Regression (S36a2 review, major 4): after an anchor move a receiver outside the previous map was lit for the whole
// blend that follows the new map's completion (the blend mixed the previous map's "none" (-1e9) with the front map's
// height). A map of 4 km at 32 m texels, 16 rows per frame: the camera moves 3 km along z (the receiver lies outside
// the first map); from the frame the new map completes on, the receiver in the ridge's shadow at sunset is shadowed
TEST_F(HorizonTest, ReceiverOutsidePreviousMapShadowedAfterAnchorMove)
{
    const Vec3 point(1500, 0, 3500);
    const Vec3 eye(1450, 120, 3500.5);
    Terrain(TerrainTestHooks::mountainRidgeHook, eye);
    const Vec3 ground(point(0), Surface(point(0), point(2)), point(2));
    LightingParameters p = scene->lightingParameters();
    p.horizonResolution  = 128;
    p.horizonExtent      = 4096;
    scene->setLightingParameters(p);
    Elevation(3);
    // The first map around z = 500: it covers z up to 2548
    Camera(Vec3(eye(0), eye(1), 500), Vec3(1, -1.5, 0));
    Capture(2);
    const uint64_t completed = scene->lightingStatistics().horizon.completed;
    p.forceFullUpdates       = false;
    p.horizonRowsPerFrame    = 16;
    scene->setLightingParameters(p);
    Camera(eye, Vec3(1, -1.5, 0));
    std::vector<float> after;
    for (int i = 0; i < 40 && after.size() < 6; ++i)
    {
        const TemplatedImage<ucvec4> img = Capture(1);
        // The statistics are taken before the frame's update: the frame that completes the map (it binds the maps of
        // the previous update) still reports the old count, the first frame that reports the new one samples the new
        // map (CodeRabbit #85 round 2: that frame was skipped too)
        const HorizonStatistics st = scene->lightingStatistics().horizon;
        if (st.completed <= completed) continue;
        after.push_back(FactorAt(img, ground));
    }
    ASSERT_EQ(after.size(), 6u) << "the new map completed";
    for (size_t i = 0; i < after.size(); ++i) EXPECT_LT(after[i], 0.1f) << "frame " << i << " after the completion";
}

// Regression (S36a2 review, finding 14): an edit during an incremental map continued that map on the new chain (its
// rows before the edit kept the old terrain; the edit showed only after a second complete map). The map in progress
// restarts: the first map completed after the edit holds the raised shadow heights behind the block
TEST_F(HorizonTest, EditRestartsTheMapInProgress)
{
    const Vec3 eye(2000, 200, 2000);
    Terrain(TerrainTestHooks::flatHook, eye);
    Elevation(10);
    Camera(eye, Vec3(1, -1, 0));
    Capture(2);
    HorizonShadows::MapTransform t;
    const std::vector<vec2> before = Front(t);
    LightingParameters p           = scene->lightingParameters();
    p.forceFullUpdates             = false;
    p.horizonRowsPerFrame          = 32;
    scene->setLightingParameters(p);
    // A new light (a jump: the base rows per frame) starts a map of 8 frames; the edit after its fifth frame (160 rows
    // done, the probed row 128 among them)
    Elevation(16);
    for (int i = 0; i < 5; ++i) RenderFrame();
    scene->waitIdle();
    ASSERT_TRUE(scene->lightingStatistics().horizon.computing);
    const uint64_t completed = scene->lightingStatistics().horizon.completed;
    std::vector<int32_t> heights(6 * 6, 100 * 1024);
    scene->patchTerrainPyramid(0, 75, 60, 6, 6, heights.data());
    for (int i = 0; i < 40 && scene->lightingStatistics().horizon.completed == completed; ++i) Capture(1);
    const HorizonStatistics st = scene->lightingStatistics().horizon;
    ASSERT_GT(st.completed, completed);
    EXPECT_EQ(st.restarts, 1u);
    HorizonShadows::MapTransform u;
    const std::vector<vec2> after = Front(u);
    const int i = int((2200.0 - u.cornerX) / u.texel), j = int((2000.0 - u.cornerZ) / u.texel);
    const size_t k = size_t(j) * size_t(u.resolution) + size_t(i);
    EXPECT_GT(after[k](0), before[k](0) + 20) << "behind the block in the first map after the edit";
}

// An edit in every frame (terraforming) restarts the map in progress only once: the map still completes (a restart
// per edit starved it: game_world --benchmark terraform never completed a map) and shows the edits of its frames
TEST_F(HorizonTest, EditsEveryFrameStillComplete)
{
    const Vec3 eye(2000, 200, 2000);
    Terrain(TerrainTestHooks::flatHook, eye);
    Elevation(10);
    Camera(eye, Vec3(1, -1, 0));
    Capture(2);
    HorizonShadows::MapTransform t0;
    const std::vector<vec2> unedited = Front(t0);
    LightingParameters p  = scene->lightingParameters();
    p.forceFullUpdates    = false;
    p.horizonRowsPerFrame = 64;
    scene->setLightingParameters(p);
    Elevation(16);
    const uint64_t completed = scene->lightingStatistics().horizon.completed;
    std::vector<int32_t> heights(6 * 6, 0);
    int frames = 0;
    for (; frames < 24 && scene->lightingStatistics().horizon.completed == completed; ++frames)
    {
        // A block from 100 m rising by 10 m per frame (every height distinct and visible behind the block at 16
        // degrees; CodeRabbit #85 round 2: from 5 m the last heights could hide behind the ground's own shadow height)
        std::fill(heights.begin(), heights.end(), (100 + frames * 10) * 1024);
        scene->patchTerrainPyramid(0, 75, 60, 6, 6, heights.data());
        Capture(1);
    }
    EXPECT_GT(scene->lightingStatistics().horizon.completed, completed) << frames << " frames";
    EXPECT_EQ(scene->lightingStatistics().horizon.restarts, 1u);
    // The edits stop: the next map (due: the chain changed) holds the block's final height, as a full recompute does
    const uint64_t first = scene->lightingStatistics().horizon.completed;
    for (int i = 0; i < 24 && scene->lightingStatistics().horizon.completed == first; ++i) Capture(1);
    ASSERT_GT(scene->lightingStatistics().horizon.completed, first);
    HorizonShadows::MapTransform t, u;
    const std::vector<vec2> next = Front(t);
    scene->requestHorizonRecompute(true);
    Capture(1);
    const std::vector<vec2> full = Front(u);
    ASSERT_EQ(next.size(), full.size());
    const int i = int((2200.0 - t.cornerX) / t.texel), j = int((2000.0 - t.cornerZ) / t.texel);
    const size_t k = size_t(j) * size_t(t.resolution) + size_t(i);
    EXPECT_GT(next[k](0), -1e8f) << "behind the block";
    // (the full recompute sees the block: the comparison can detect a stale map)
    EXPECT_GT(full[k](0), unedited[k](0) + 20) << "behind the block";
    EXPECT_EQ(next[k](0), full[k](0)) << "behind the block";
}

// A terrain edit (a pyramid patch) rebuilds the max chain and recomputes the whole map (nit 20: the name said dirty
// rows): a raised block towards the light raises the shadow heights behind it
TEST_F(HorizonTest, EditRecomputesMap)
{
    const Vec3 eye(2000, 200, 2000);
    Terrain(TerrainTestHooks::flatHook, eye);
    Elevation(10);
    Camera(eye, Vec3(1, -1, 0));
    Capture(2);
    HorizonShadows::MapTransform t;
    const std::vector<vec2> before = Front(t);
    const uint64_t recomputes      = scene->lightingStatistics().horizon.recomputes;
    // A 100 m block at x 2400..2560 (pyramid samples 75..80), z 1920..2080
    std::vector<int32_t> heights(6 * 6, 100 * 1024);
    scene->patchTerrainPyramid(0, 75, 60, 6, 6, heights.data());
    Capture(3);
    EXPECT_GT(scene->lightingStatistics().horizon.recomputes, recomputes);
    HorizonShadows::MapTransform u;
    const std::vector<vec2> after = Front(u);
    const int i = int((2200.0 - u.cornerX) / u.texel), j = int((2000.0 - u.cornerZ) / u.texel);
    const size_t k = size_t(j) * size_t(u.resolution) + size_t(i);
    EXPECT_GT(after[k](0), before[k](0) + 20) << "behind the block";
    const int jFar    = int((2600.0 - u.cornerZ) / u.texel);
    const size_t kFar = size_t(jFar) * size_t(u.resolution) + size_t(i);
    EXPECT_NEAR(after[kFar](0), before[kFar](0), 1e-3) << "beside it";
}

// Regression (CodeRabbit #85 round 2): after clearTerrain the front map of the removed terrain stayed valid and shown
// in its space (the update returned before it invalidated the maps)
TEST_F(HorizonTest, ClearedTerrainHidesTheMap)
{
    const Vec3 eye(1450, 120, 2048.5);
    Terrain(TerrainTestHooks::mountainRidgeHook, eye);
    Elevation(3);
    Camera(eye, Vec3(1, -1.5, 0));
    Capture(2);
    ASSERT_TRUE(scene->lightingStatistics().horizon.valid);
    scene->clearTerrain();
    Capture(2);
    EXPECT_FALSE(scene->lightingStatistics().horizon.valid);
    const LightingParameters p = scene->lightingParameters();
    SceneLighting block;
    scene->horizonShadows().fillBlock(block, eye, 0, p, true, true);
    EXPECT_LT(block.horizonFront(3), 0.0f) << "no map after the terrain was cleared";
}

// Regression (CodeRabbit #85 round 2): a full update cancelled the map in progress, but started none when nothing else
// was due (the edit's request was taken by the cancelled map): the front map kept the terrain before the edit
TEST_F(HorizonTest, FullUpdateReplacesTheMapInProgress)
{
    const Vec3 eye(2000, 200, 2000);
    Terrain(TerrainTestHooks::flatHook, eye);
    Elevation(10);
    Camera(eye, Vec3(1, -1, 0));
    Capture(2);
    HorizonShadows::MapTransform t;
    const std::vector<vec2> before = Front(t);
    LightingParameters p           = scene->lightingParameters();
    p.forceFullUpdates             = false;
    p.horizonRowsPerFrame          = 32;
    scene->setLightingParameters(p);
    // The edit starts a map of 8 frames; after two of them a full update
    std::vector<int32_t> heights(6 * 6, 100 * 1024);
    scene->patchTerrainPyramid(0, 75, 60, 6, 6, heights.data());
    RenderFrame();
    RenderFrame();
    scene->waitIdle();
    ASSERT_TRUE(scene->horizonShadows().statistics().computing);
    const uint64_t completed = scene->horizonShadows().statistics().completed;
    // (the scene's one-shot full update: unlike requestHorizonRecompute it requests no map of its own)
    scene->requestFullUpdate();
    RenderFrame();
    scene->waitIdle();
    const HorizonStatistics st = scene->horizonShadows().statistics();
    EXPECT_GT(st.completed, completed) << "the replacement completed in the frame of the full update";
    HorizonShadows::MapTransform u;
    const std::vector<vec2> after = Front(u);
    const int i = int((2200.0 - u.cornerX) / u.texel), j = int((2000.0 - u.cornerZ) / u.texel);
    const size_t k = size_t(j) * size_t(u.resolution) + size_t(i);
    EXPECT_GT(after[k](0), before[k](0) + 20) << "behind the block";
}

// Regression (CodeRabbit #85 round 2): after the one restart of a map, a second edit rebuilt the chain under the map in
// progress, so its rows mixed two terrain versions. The map keeps the terrain of its restart: it equals a full
// recompute of that terrain, not of the second edit's higher block
TEST_F(HorizonTest, LaterEditWaitsForTheNextMap)
{
    const Vec3 eye(2000, 200, 2000);
    Terrain(TerrainTestHooks::flatHook, eye);
    Elevation(10);
    Camera(eye, Vec3(1, -1, 0));
    Capture(2);
    LightingParameters p  = scene->lightingParameters();
    p.forceFullUpdates    = false;
    p.horizonRowsPerFrame = 32;
    scene->setLightingParameters(p);
    // A new light starts a map; the first edit restarts it, the second (a higher block) comes before the probed row
    // 128 is computed
    Elevation(16);
    RenderFrame();
    std::vector<int32_t> low(6 * 6, 60 * 1024), high(6 * 6, 200 * 1024);
    scene->patchTerrainPyramid(0, 75, 60, 6, 6, low.data());
    RenderFrame();
    scene->patchTerrainPyramid(0, 75, 60, 6, 6, high.data());
    const uint64_t completed = scene->horizonShadows().statistics().completed;
    for (int i = 0; i < 40 && scene->horizonShadows().statistics().completed == completed; ++i)
    {
        RenderFrame();
        scene->waitIdle();
    }
    ASSERT_GT(scene->horizonShadows().statistics().completed, completed);
    EXPECT_EQ(scene->horizonShadows().statistics().restarts, 1u);
    HorizonShadows::MapTransform t, u;
    const std::vector<vec2> restarted = Front(t);
    // The full recompute of the restart's terrain
    scene->patchTerrainPyramid(0, 75, 60, 6, 6, low.data());
    scene->requestHorizonRecompute(true);
    Capture(1);
    const std::vector<vec2> full = Front(u);
    ASSERT_EQ(restarted.size(), full.size());
    const int i = int((2200.0 - t.cornerX) / t.texel), j = int((2000.0 - t.cornerZ) / t.texel);
    const size_t k = size_t(j) * size_t(t.resolution) + size_t(i);
    EXPECT_GT(full[k](0), -1e8f) << "behind the block";
    EXPECT_NEAR(restarted[k](0), full[k](0), 1e-3) << "behind the block: the restart's terrain";
}
