/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The local lights of S36b on RenderScene (docs/stages/S36_lighting.plan.md, B2-B4, Tests): the light selection and
// budget, the falloff and cones, flicker, the per-chunk lists, enclosed spaces; the clusters against a CPU reference
// and the brute-force loop; the cached shadow atlas. One scene per test suite (RenderSceneSuiteTest), skipped without a
// Vulkan 1.3 device.

#include "saiga/vulkan/scene/LightClusters.h"
#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/LocalLights.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/ShadowAtlas.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>

#include "render_scene_test_context.h"

using namespace Saiga;
using namespace Saiga::Vulkan;

namespace
{
constexpr double kPi = 3.14159265358979323846;

Quat lookRotation(const Vec3& forward)
{
    const Vec3 back  = -forward.normalized();
    const Vec3 right = (std::abs(back(1)) > 0.999 ? Vec3(0, 0, -1) : Vec3(0, 1, 0)).cross(back).normalized();
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

MeshLods boxMesh(const vec3& half, const ucvec4& color)
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
    MeshLod lod;
    lod.endDistance = 2000;
    for (const Face& f : faces)
    {
        const uint32_t base = uint32_t(lod.vertices.size());
        for (const vec3& c : f.c)
        {
            MeshVertex v;
            v.position = c.cwiseProduct(half);
            v.normal   = packNormalOct(f.n);
            v.color    = packColor(color);
            lod.vertices.push_back(v);
        }
        for (uint32_t i : {0u, 1u, 2u, 0u, 2u, 3u}) lod.indices.push_back(base + i);
    }
    MeshLods m;
    m.lods.push_back(lod);
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

// No sun, moon, fog or ambient floor: only the local lights light the scene (EV100 fixed)
EnvironmentState nightEnvironment(float ev100 = 2)
{
    EnvironmentState e;
    e.sunIlluminance  = 0;
    e.moonIlluminance = 0;
    e.fogDensity      = 0;
    e.minAmbient      = vec3(0, 0, 0);
    e.ambientScale    = 0;
    e.starIntensity   = 0;
    e.cloudiness      = 0;
    e.ev100           = ev100;
    return e;
}

LightData pointLight(const vec3& position, float lumen, float range)
{
    LightData d;
    d.kind     = LightKind::Point;
    d.position = position;
    d.lumen    = lumen;
    d.range    = range;
    d.color    = vec3(1, 1, 1);
    return d;
}

LightData spotLight(const vec3& position, const vec3& direction, float lumen, float range, float innerDeg,
                    float outerDeg)
{
    LightData d = pointLight(position, lumen, range);
    d.kind      = LightKind::Spot;
    d.direction = direction.normalized();
    d.cosInner  = float(std::cos(innerDeg * kPi / 180));
    d.cosOuter  = float(std::cos(outerDeg * kPi / 180));
    return d;
}

class LightsTest : public RenderSceneSuiteTest
{
   protected:
    void SetUp() override
    {
        RenderSceneSuiteTest::SetUp();
        if (IsSkipped()) return;
        for (LightHandle h : lights) scene->removeLight(h);
        lights.clear();
        LightingParameters p;
        p.localLights     = true;
        p.clusteredLights = false;
        p.sunShadows      = false;
        p.horizonShadows  = false;
        p.clouds          = false;
        scene->setLightingParameters(p);
        scene->setEnvironment(nightEnvironment());
        scene->setToneMap(ToneMap::Aces);
        scene->setLightingDebug(LightingDebugState());
    }
    void TearDown() override
    {
        if (scene)
        {
            for (LightHandle h : lights) scene->removeLight(h);
            lights.clear();
            scene->setLightingDebug(LightingDebugState());
        }
        RenderSceneSuiteTest::TearDown();
    }
    LightHandle AddLight(const LightData& d, const ChunkId& chunk = ChunkId())
    {
        lights.push_back(scene->addLight(chunk, d));
        return lights.back();
    }
    // A gray floor (top at y = 0, 80 m wide) in space 0, the camera above it looking down
    void Floor(MeshId first, const Vec3& eye = Vec3(0, 12, 0.01))
    {
        MaterialData m;
        m.roughness = 1.0f;
        scene->addMaterial(MaterialId(first), "light_floor" + std::to_string(first), m);
        scene->addMesh(first, "light_floor" + std::to_string(first),
                       boxMesh(vec3(40, 0.05f, 40), ucvec4(160, 160, 160, 255)), MaterialId(first));
        AddSet({instanceAt(vec3(0, -0.05f, 0), first)});
        SetCamera(0, eye, lookRotation(Vec3(0, -1, -0.001)), 1.2);
    }
    // The resolved fp16 value of a pixel in the next frame, divided by the frame's pre-exposure: the radiance
    vec3 ReadRadiance(ivec2 p)
    {
        vec4 value(-1, -1, -1, -1);
        scene->lightingDebugImages().requestPixels({p},
                                                   [&](const std::vector<LightingDebug::PixelValue>& v)
                                                   {
                                                       if (!v.empty() && v[0].valid) value = v[0].color;
                                                   });
        RenderFrame();
        const float pe = scene->preExposure();
        return vec3(value(0), value(1), value(2)) / pe;
    }
    // The screen pixel of a world point (the frame's camera-relative matrices)
    ivec2 Pixel(const Vec3& world, const Vec3& camera)
    {
        const Vec3 rel = world - camera;
        const vec4 h   = scene->viewProjection() * vec4(float(rel(0)), float(rel(1)), float(rel(2)), 1);
        return ivec2(int((h(0) / h(3) * 0.5f + 0.5f) * float(width)), int((h(1) / h(3) * 0.5f + 0.5f) * float(height)));
    }
    std::vector<LightHandle> lights;
};
}  // namespace

// The Lambert part of a point light 2 m above a rough floor: radiance albedo / pi * I / d^2 * window, within 25 %
// (the GGX specular of roughness 1 and the Fresnel add a little); off: black
TEST_F(LightsTest, PointLightFollowsInverseSquare)
{
    Floor(700);
    AddLight(pointLight(vec3(0, 2, 0), 1200, 18));
    Capture(3);
    const Vec3 eye(0, 12, 0.01);
    const vec3 centre   = ReadRadiance(Pixel(Vec3(0, 0, 0), eye));
    const double albedo = std::pow((160.0 / 255 + 0.055) / 1.055, 2.4);
    const double I      = 1200 / (4 * kPi);
    const double w      = std::pow(1 - std::pow(2.0 / 18, 4), 2);
    const double L      = albedo / kPi * I / 4 * w;
    EXPECT_NEAR(centre(1), L, 0.25 * L) << centre.transpose();
    // 6 m away (oblique: cos 18.4 deg at the floor, d^2 = 40): darker
    const vec3 side = ReadRadiance(Pixel(Vec3(6, 0, 0), eye));
    EXPECT_LT(side(1), centre(1) * 0.2f);
    const LocalLightStatistics& st = scene->lightingStatistics().lights;
    EXPECT_EQ(st.visible, 1u);
    EXPECT_EQ(st.submitted, 1u);
    LightingParameters p = scene->lightingParameters();
    p.localLights        = false;
    scene->setLightingParameters(p);
    Capture(2);
    EXPECT_LT(ReadRadiance(Pixel(Vec3(0, 0, 0), eye))(1), 1e-3f * float(L));
    EXPECT_EQ(scene->lightingStatistics().lights.visible, 0u);
}

// The budget keeps the most important lights (priority * lumen / max(distance, range)^2, ties by handle) and counts
// the rest
TEST_F(LightsTest, BudgetKeepsMostImportant)
{
    Floor(702);
    std::vector<LightHandle> h;
    for (int i = 0; i < 10; ++i) h.push_back(AddLight(pointLight(vec3(float(i) - 5, 1, 0), 100.0f * float(i + 1), 4)));
    // Two equal ones: the lower handle wins
    LightData tie        = pointLight(vec3(0, 1, 3), 2000, 4);
    const LightHandle t1 = AddLight(tie);
    const LightHandle t2 = AddLight(tie);
    LightingParameters p = scene->lightingParameters();
    p.maxLights          = 4;
    scene->setLightingParameters(p);
    Capture(2);
    const std::vector<FrameLightInfo>& f = scene->frameLights();
    ASSERT_EQ(f.size(), 4u);
    EXPECT_EQ(f[0].handle, t1.id);
    EXPECT_EQ(f[1].handle, t2.id);
    EXPECT_EQ(f[2].handle, h[9].id);
    EXPECT_EQ(f[3].handle, h[8].id);
    EXPECT_EQ(scene->lightingStatistics().lights.overBudget, 8u);
    EXPECT_EQ(scene->lightingStatistics().lights.visible, 4u);
    for (size_t i = 1; i < f.size(); ++i) EXPECT_GE(f[i - 1].importance, f[i].importance);
    // Behind the camera or beyond clusterFar + range: not a candidate
    AddLight(pointLight(vec3(0, 40, 0), 5000, 4));
    Capture(1);
    EXPECT_EQ(scene->lightingStatistics().lights.candidates, 12u);
}

// A spot light pointing down: the floor inside the inner cone is lit, outside the outer cone dark
TEST_F(LightsTest, SpotCone)
{
    Floor(704);
    AddLight(spotLight(vec3(0, 4, 0), vec3(0, -1, 0), 4000, 20, 20, 30));
    Capture(3);
    const Vec3 eye(0, 12, 0.01);
    const vec3 inside = ReadRadiance(Pixel(Vec3(0.5, 0, 0), eye));
    // tan(30 deg) * 4 = 2.31 m: 3.5 m is outside
    const vec3 outside = ReadRadiance(Pixel(Vec3(3.5, 0, 0), eye));
    EXPECT_GT(inside(1), 0.0f);
    EXPECT_LT(outside(1), inside(1) * 1e-3f) << inside.transpose() << " / " << outside.transpose();
}

// Flicker is a function of the environment's flicker phase: the same phase renders the same image, another phase
// another one
TEST_F(LightsTest, FlickerUsesPhases)
{
    Floor(706);
    LightData d         = pointLight(vec3(0, 2, 0), 1200, 18);
    d.flicker.amplitude = 0.5f;
    d.flicker.frequency = 3;
    d.flicker.seed      = 17;
    AddLight(d);
    const Vec3 eye(0, 12, 0.01);
    auto at = [&](float phase)
    {
        EnvironmentState e = nightEnvironment();
        e.phases[6]        = phase;
        scene->setEnvironment(e);
        Capture(2);
        return ReadRadiance(Pixel(Vec3(0, 0, 0), eye))(1);
    };
    const float a = at(0.7f), b = at(2.1f), c = at(0.7f);
    EXPECT_EQ(a, c);
    EXPECT_GT(std::abs(a - b), 0.02f * a);
    EXPECT_NEAR(LocalLights::flickerFactor(d.flicker, 0.0f), LocalLights::flickerFactor(d.flicker, float(2 * kPi)),
                1e-4f);
}

// addLights / removeLights: one list per chunk (S30's piece lights), positions relative to the chunk
TEST_F(LightsTest, BulkListPerChunk)
{
    Floor(708);
    std::vector<LightData> a = {pointLight(vec3(1, 1, 1), 500, 6), pointLight(vec3(3, 1, 1), 500, 6)};
    std::vector<LightData> b = {pointLight(vec3(250, 1, 2), 500, 6)};
    // Chunk (-1, 0): x = 250 - 256 = -6 m in the world
    const LightListHandle la = scene->addLights(ChunkId{0, 0, 0}, a);
    const LightListHandle lb = scene->addLights(ChunkId{0, -1, 0}, b);
    EXPECT_EQ(scene->localLights().size(), 3u);
    EXPECT_EQ(scene->localLights().listCount(), 2u);
    Capture(2);
    EXPECT_EQ(scene->frameLights().size(), 3u);
    bool found = false;
    for (const LocalLights::Light& l : scene->localLights().lights())
        if (l.list == lb.id)
        {
            EXPECT_NEAR((l.worldPosition() - Vec3(-6, 1, 2)).norm(), 0, 1e-5);
            found = true;
        }
    EXPECT_TRUE(found);
    // The record is camera-relative
    const Vec3 eye(0, 12, 0.01);
    for (const FrameLightInfo& f : scene->frameLights())
    {
        const LocalLights::Light* l = scene->localLights().find(LightHandle{f.handle});
        ASSERT_TRUE(l);
        const vec4 r = scene->localLights().frameRecords()[f.index].posRange;
        EXPECT_NEAR((Vec3(r(0), r(1), r(2)) - (l->worldPosition() - eye)).norm(), 0, 1e-4);
    }
    scene->removeLights(la);
    Capture(1);
    EXPECT_EQ(scene->localLights().size(), 1u);
    EXPECT_EQ(scene->frameLights().size(), 1u);
    scene->removeLights(lb);
    Capture(1);
    EXPECT_EQ(scene->frameLights().size(), 0u);
}

// An enclosed space (a dungeon) is lit by its own lights; a light of another space does not reach it
TEST_F(LightsTest, EnclosedSpaceLit)
{
    MaterialData m;
    m.roughness = 1.0f;
    scene->addMaterial(MaterialId(710), "enclosed_floor", m);
    scene->addMesh(710, "enclosed_floor", boxMesh(vec3(20, 0.05f, 20), ucvec4(160, 160, 160, 255)), MaterialId(710));
    RegionId room;
    room.space = 3;
    sets.push_back(scene->addInstanceSet(room, {instanceAt(vec3(0, -0.05f, 0), 710)}));
    SpaceRenderSettings enclosed;
    enclosed.openWorld = false;
    scene->setSpaceSettings(3, enclosed);
    EnvironmentState e = nightEnvironment(4);
    e.enclosed         = true;
    e.enclosedAmbient  = vec3(0, 0, 0);
    scene->setEnvironment(e);
    const Vec3 eye(0, 8, 0.01);
    SetCamera(3, eye, lookRotation(Vec3(0, -1, -0.001)), 1.2);
    AddLight(pointLight(vec3(0, 1, 0), 600, 8), ChunkId{0, 0, 0});  // space 0: not in the room
    Capture(2);
    EXPECT_EQ(scene->frameLights().size(), 0u);
    EXPECT_LT(ReadRadiance(Pixel(Vec3(0, 0, 0), eye))(1), 1e-4f);
    ChunkId roomChunk;
    roomChunk.space = 3;
    AddLight(pointLight(vec3(0, 1.5f, 0), 600, 8), roomChunk);
    Capture(2);
    EXPECT_EQ(scene->frameLights().size(), 1u);
    EXPECT_GT(ReadRadiance(Pixel(Vec3(0, 0, 0), eye))(1), 1.0f);
    scene->setSpaceSettings(3, SpaceRenderSettings());
}

// The LocalLightsOnly view and the LightVolumes overlay (with a highlighted light) render without a validation error
TEST_F(LightsTest, DebugViewAndVolumes)
{
    Floor(712);
    const LightHandle a = AddLight(pointLight(vec3(-2, 1, 0), 800, 6));
    LightData s         = spotLight(vec3(2, 3, 0), vec3(0, -1, 0), 2000, 8, 20, 30);
    s.castShadows       = true;
    AddLight(s);
    scene->setLightingDebug(LightingDebugState{LightingDebugView::LocalLightsOnly, OverlayLightVolumes, a.id});
    const TemplatedImage<ucvec4> img = Capture(2);
    EXPECT_EQ(scene->localLights().find(a) != nullptr, true);
    double sum = 0;
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) sum += img(y, x)(1);
    EXPECT_GT(sum, 0);
}

namespace
{
// 300 random point and spot lights over the floor (deterministic)
void randomLights(std::vector<LightData>& out, int n, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0, 1);
    for (int i = 0; i < n; ++i)
    {
        const vec3 p(-30 + 60 * u(rng), 0.3f + 3 * u(rng), -30 + 60 * u(rng));
        LightData d = u(rng) < 0.7f ? pointLight(p, 50 + 600 * u(rng), 1 + 6 * u(rng))
                                    : spotLight(p, vec3(u(rng) - 0.5f, -1, u(rng) - 0.5f), 200 + 1500 * u(rng),
                                                2 + 8 * u(rng), 10 + 15 * u(rng), 30 + 20 * u(rng));
        d.color     = vec3(0.3f + 0.7f * u(rng), 0.3f + 0.7f * u(rng), 0.3f + 0.7f * u(rng));
        out.push_back(d);
    }
}
}  // namespace

// The LocalLightsOnly image with clusters equals the brute-force loop within 1/255 (300 random lights; enough slots:
// no overflow)
TEST_F(LightsTest, ClusteredEqualsBruteForce)
{
    Floor(720, Vec3(0, 18, 26));
    SetCamera(0, Vec3(0, 18, 26), lookRotation(Vec3(0, -0.8, -1)), 1.2);
    std::vector<LightData> list;
    randomLights(list, 300, 7);
    for (const LightData& d : list) AddLight(d);
    LightingParameters p  = scene->lightingParameters();
    p.maxLightsPerCluster = 512;
    p.clusteredLights     = false;
    scene->setLightingParameters(p);
    scene->setLightingDebug(LightingDebugState{LightingDebugView::LocalLightsOnly, 0});
    const TemplatedImage<ucvec4> brute = Capture(3);
    p.clusteredLights                  = true;
    scene->setLightingParameters(p);
    const TemplatedImage<ucvec4> clustered = Capture(3);
    EXPECT_EQ(scene->lightingStatistics().lights.overflowedClusters, 0u);
    EXPECT_TRUE(scene->lightingStatistics().lights.clustered);
    int maxDiff = 0;
    double lit  = 0;
    for (int y = 0; y < brute.h; ++y)
        for (int x = 0; x < brute.w; ++x)
            for (int c = 0; c < 3; ++c)
            {
                maxDiff = std::max(maxDiff, std::abs(int(brute(y, x)(c)) - int(clustered(y, x)(c))));
                lit += brute(y, x)(c);
            }
    EXPECT_LE(maxDiff, 1);
    EXPECT_GT(lit / (brute.w * brute.h * 3), 5.0) << "the lights light the floor";
}

// Per cluster the GPU's light lists equal a CPU assignment (in importance order); a light that differs lies on the
// cluster's boundary (within 1e-3 of its range)
TEST_F(LightsTest, ClusterListsMatchCpuReference)
{
    Floor(722, Vec3(0, 18, 26));
    SetCamera(0, Vec3(0, 18, 26), lookRotation(Vec3(0, -0.8, -1)), 1.2);
    std::vector<LightData> list;
    randomLights(list, 300, 11);
    for (const LightData& d : list) AddLight(d);
    LightingParameters p  = scene->lightingParameters();
    p.clusteredLights     = true;
    p.maxLightsPerCluster = 64;
    scene->setLightingParameters(p);
    Capture(2);
    LightClusters& lc                                 = scene->lightClusters();
    const uint32_t slot                               = lc.lastRecordedSlot();
    const std::vector<uint32_t> table                 = lc.readTableForTests(slot);
    const std::vector<uint32_t> indices               = lc.readIndicesForTests(slot);
    const ClusterGrid& grid                           = scene->frameClusterGrid();
    const std::vector<LocalLights::GpuLight>& records = scene->localLights().frameRecords();
    ASSERT_GT(records.size(), 200u);
    const std::vector<std::vector<uint32_t>> cpu =
        LightClusters::computeCpu(grid, scene->viewMatrix(), scene->projectionMatrix(), records);
    auto borderline = [&](uint32_t i, int c)
    {
        vec3 lo, hi;
        LightClusters::clusterBounds(grid, scene->projectionMatrix(), c, lo, hi);
        LocalLights::GpuLight a = records[i], b = records[i];
        a.posRange(3) *= 1.001f;
        b.posRange(3) *= 0.999f;
        a.cone(0) -= 1e-3f;
        b.cone(0) += 1e-3f;
        return LightClusters::lightTouches(a, scene->viewMatrix(), lo, hi) !=
               LightClusters::lightTouches(b, scene->viewMatrix(), lo, hi);
    };
    int nonEmpty = 0, differences = 0, overflowed = 0;
    for (int c = 0; c < grid.count(); ++c)
    {
        const uint32_t count = table[2 * size_t(c)];
        const uint32_t total = table[2 * size_t(c) + 1];
        nonEmpty += count > 0 ? 1 : 0;
        overflowed += total > count ? 1 : 0;
        std::vector<uint32_t> gpu(indices.begin() + ptrdiff_t(size_t(c) * 64),
                                  indices.begin() + ptrdiff_t(size_t(c) * 64 + count));
        // In importance order: increasing buffer indices
        EXPECT_TRUE(std::is_sorted(gpu.begin(), gpu.end())) << "cluster " << c;
        if (total > count) continue;  // the cap's cut can differ by a borderline light: checked below
        std::set<uint32_t> a(gpu.begin(), gpu.end()), b(cpu[size_t(c)].begin(), cpu[size_t(c)].end());
        for (uint32_t i : a)
            if (!b.count(i))
            {
                differences++;
                EXPECT_TRUE(borderline(i, c)) << "light " << i << " only on the GPU in cluster " << c;
            }
        for (uint32_t i : b)
            if (!a.count(i))
            {
                differences++;
                EXPECT_TRUE(borderline(i, c)) << "light " << i << " only on the CPU in cluster " << c;
            }
    }
    EXPECT_GT(nonEmpty, 100);
    EXPECT_LT(differences, 20);
    EXPECT_EQ(overflowed, 0);
}

// A cluster with more lights than slots keeps the most important ones and counts the overflow (statistics of the
// completed frame)
TEST_F(LightsTest, OverflowCounted)
{
    Floor(724);
    for (int i = 0; i < 12; ++i) AddLight(pointLight(vec3(0.1f * float(i), 1, 0), 100.0f + 10.0f * float(i), 6));
    LightingParameters p  = scene->lightingParameters();
    p.clusteredLights     = true;
    p.maxLightsPerCluster = 4;
    scene->setLightingParameters(p);
    Capture(4);
    const LocalLightStatistics& st = scene->lightingStatistics().lights;
    EXPECT_GT(st.overflowedClusters, 0u);
    EXPECT_GT(st.droppedLights, 0u);
    EXPECT_EQ(st.clusterMax, 4u);
    // The slots hold the four most important ones: buffer indices 0..3
    LightClusters& lc                   = scene->lightClusters();
    const uint32_t slot                 = lc.lastRecordedSlot();
    const std::vector<uint32_t> table   = lc.readTableForTests(slot);
    const std::vector<uint32_t> indices = lc.readIndicesForTests(slot);
    for (int c = 0; c < scene->frameClusterGrid().count(); ++c)
        if (table[2 * size_t(c)] == 4 && table[2 * size_t(c) + 1] == 12)
        {
            for (uint32_t k = 0; k < 4; ++k) EXPECT_EQ(indices[size_t(c) * 4 + k], k);
            return;
        }
    ADD_FAILURE() << "no cluster with all 12 lights";
}

// The ClusterLights view, the ClusterGrid overlay and the Clusters preview render (the preview's maximum equals the
// statistics' maximum)
TEST_F(LightsTest, ClusterDebug)
{
    Floor(726);
    for (int i = 0; i < 6; ++i) AddLight(pointLight(vec3(float(i) - 3, 1, 0), 400, 4));
    LightingParameters p = scene->lightingParameters();
    p.clusteredLights    = true;
    scene->setLightingParameters(p);
    scene->setLightingDebug(LightingDebugState{LightingDebugView::ClusterLights, OverlayClusterGrid});
    Capture(3);
    PreviewMapping m;
    PreviewStatistics st;
    bool got = false;
    scene->setClusterPreviewSlice(-1);
    scene->lightingDebugImages().request(LightingImage::Clusters, m,
                                         [&](const LightingDebug::Capture& c)
                                         {
                                             st  = c.statistics;
                                             got = true;
                                             EXPECT_EQ(c.size(0), 16 * LightClusters::kCellPixels);
                                         });
    Capture(2);
    ASSERT_TRUE(got);
    EXPECT_GT(st.max, 1.0f);
    EXPECT_LE(st.max, 6.0f);
}

namespace
{
// The atlas suite: the light tests' scene with a small atlas (1024 texels in tiles of 128: 64 tiles as at the
// defaults; lavapipe)
class AtlasTest : public LightsTest
{
   protected:
    void SetUp() override
    {
        LightsTest::SetUp();
        if (IsSkipped()) return;
        LightingParameters p = scene->lightingParameters();
        p.clusteredLights    = true;
        p.atlasResolution    = 1024;
        p.atlasTile          = 128;
        p.facesPerFrame      = 11;
        scene->setLightingParameters(p);
    }
    void SetFaces(int faces)
    {
        LightingParameters p = scene->lightingParameters();
        p.facesPerFrame      = faces;
        scene->setLightingParameters(p);
    }
    const AtlasStatistics& Stats() { return scene->lightingStatistics().atlas; }
    // A floor and a box (1 m, top at 1 m) at x, z
    void FloorAndBox(MeshId first, const vec3& box = vec3(0, 0.5f, 0))
    {
        Floor(first);
        MaterialData m;
        scene->addMaterial(MaterialId(first + 1), "atlas_box" + std::to_string(first), m);
        scene->addMesh(MeshId(first + 1), "atlas_box" + std::to_string(first),
                       boxMesh(vec3(0.5f, 0.5f, 0.5f), ucvec4(200, 200, 200, 255)), MaterialId(first + 1));
        boxSet    = AddSet({instanceAt(box, MeshId(first + 1))});
        boxMeshId = MeshId(first + 1);
    }
    LightData ShadowedPoint(const vec3& p, float range = 8)
    {
        LightData d   = pointLight(p, 2000, range);
        d.castShadows = true;
        return d;
    }
    InstanceSetHandle boxSet;
    MeshId boxMeshId = 0;
};
}  // namespace

// A static scene: the faces are rendered once; then every face is a cache hit (no re-render)
TEST_F(AtlasTest, StaticSceneCachesFaces)
{
    FloorAndBox(740);
    AddLight(ShadowedPoint(vec3(2, 3, 0)));
    RenderFrame();
    EXPECT_EQ(Stats().staticRenders, 6u);
    EXPECT_EQ(Stats().tilesUsed, 6u);
    EXPECT_EQ(Stats().tilesTotal, 64u);
    Capture(2);
    EXPECT_EQ(Stats().staticRenders, 0u);
    EXPECT_EQ(Stats().hits, 6u);
    EXPECT_EQ(Stats().queue, 0u);
    const std::vector<FrameLightInfo>& f = scene->frameLights();
    ASSERT_EQ(f.size(), 1u);
    EXPECT_EQ(f[0].tiles, 6);
    EXPECT_TRUE(f[0].shadowReady);
}

// A light that moves re-renders its faces (its key changes); another light's stay cached
TEST_F(AtlasTest, MovingLightRerendersItsFaces)
{
    FloorAndBox(742);
    const LightHandle a = AddLight(ShadowedPoint(vec3(2, 3, 0)));
    LightData spot      = spotLight(vec3(-3, 4, 0), vec3(0, -1, 0), 3000, 10, 20, 35);
    spot.castShadows    = true;
    AddLight(spot);
    Capture(2);
    EXPECT_EQ(Stats().staticRenders, 0u);
    scene->setLight(a, ChunkId(), ShadowedPoint(vec3(2.5f, 3, 0)));
    RenderFrame();
    EXPECT_EQ(Stats().staticRenders, 6u);
    EXPECT_EQ(Stats().hits, 1u) << "the spot's tile";
}

// An instance change inside a light's range re-renders its faces, one outside does not
TEST_F(AtlasTest, InstanceChangeInRangeRerenders)
{
    FloorAndBox(744);
    AddLight(ShadowedPoint(vec3(0, 3, 0), 6));
    // A far set (60 m away, outside the range)
    const InstanceSetHandle far = AddSet({instanceAt(vec3(30, 0.5f, 30), boxMeshId)});
    Capture(2);
    EXPECT_EQ(Stats().staticRenders, 0u);
    scene->updateInstances(far, 0, {instanceAt(vec3(31, 0.5f, 30), boxMeshId)});
    RenderFrame();
    EXPECT_EQ(Stats().staticRenders, 0u) << "outside the range";
    scene->updateInstances(boxSet, 0, {instanceAt(vec3(1, 0.5f, 0), boxMeshId)});
    RenderFrame();
    EXPECT_EQ(Stats().staticRenders, 6u) << "inside";
    RenderFrame();
    scene->removeInstance(boxSet, 0);
    RenderFrame();
    EXPECT_EQ(Stats().staticRenders, 6u) << "a tombstone inside";
    // A set leaving the range (its old bounds) and another changing outside in the same frame
    RenderFrame();
    scene->updateInstances(far, 0, {instanceAt(vec3(1, 0.5f, 1), boxMeshId)});
    RenderFrame();
    EXPECT_EQ(Stats().staticRenders, 6u) << "a set moving into the range";
    RenderFrame();
    scene->updateInstances(far, 0, {instanceAt(vec3(40, 0.5f, 40), boxMeshId)});
    RenderFrame();
    EXPECT_EQ(Stats().staticRenders, 6u) << "a set leaving the range";
}

// A camera move keeps the static tiles (their content is camera-independent)
TEST_F(AtlasTest, CameraMoveKeepsStaticTiles)
{
    FloorAndBox(746);
    AddLight(ShadowedPoint(vec3(0, 3, 0)));
    Capture(2);
    SetCamera(0, Vec3(3, 10, 4), lookRotation(Vec3(-0.3, -1, -0.4)), 1.2);
    Capture(2);
    EXPECT_EQ(Stats().staticRenders, 0u);
    EXPECT_EQ(Stats().hits, 6u);
}

// The dynamic layer is rendered only for the lights with a dynamic instance in range, and cleared once when the last
// one leaves
TEST_F(AtlasTest, DynamicLayerOnlyNearDynamics)
{
    FloorAndBox(748);
    AddLight(ShadowedPoint(vec3(0, 3, 0), 6));
    LightData spot   = spotLight(vec3(20, 4, 0), vec3(0, -1, 0), 3000, 6, 20, 35);
    spot.castShadows = true;
    AddLight(spot);
    const InstanceHandle d = scene->addDynamic(boxMeshId);
    DynamicPose pose;
    pose.position = Vec3(1, 1.5, 0);
    scene->setDynamic(d, ChunkId(), pose, pose);
    Capture(2);
    EXPECT_EQ(Stats().dynamicRenders, 6u) << "the point light's faces only";
    int dynamicTiles = 0;
    for (const AtlasTileInfo& t : scene->shadowAtlas().tiles()) dynamicTiles += t.dynamicContent ? 1 : 0;
    EXPECT_EQ(dynamicTiles, 6);
    // Away from both lights: the dynamic tiles are cleared once
    pose.position = Vec3(-40, 1.5, 0);
    scene->setDynamic(d, ChunkId(), pose, pose);
    RenderFrame();
    EXPECT_EQ(Stats().dynamicRenders, 0u);
    EXPECT_EQ(Stats().dynamicClears, 6u);
    RenderFrame();
    EXPECT_EQ(Stats().dynamicClears, 0u);
    scene->removeDynamic(d);
}

// The tile budget: 16 shadowed point lights in view, 64 tiles: 10 get tiles, 6 are counted without
TEST_F(AtlasTest, TileBudgetLimitsShadowedLights)
{
    Floor(750);
    for (int i = 0; i < 16; ++i) AddLight(ShadowedPoint(vec3(-7.5f + 1.0f * float(i), 2, 0), 4 + 0.01f * float(i)));
    Capture(2);
    EXPECT_EQ(Stats().lightsWithTiles, 10u);
    EXPECT_EQ(Stats().lightsWithoutTiles, 6u);
    EXPECT_EQ(Stats().tilesUsed, 60u);
    EXPECT_EQ(scene->lightingStatistics().lights.withoutTiles, 6u);
}

// facesPerFrame limits the renders; the queue is worked off by importance; a light is drawn unshadowed until each of
// its faces was rendered
TEST_F(AtlasTest, BudgetAndQueueOrder)
{
    Floor(752);
    SetFaces(4);
    const LightHandle weak   = AddLight(ShadowedPoint(vec3(-4, 2, 0), 4));
    LightData strongData     = ShadowedPoint(vec3(4, 2, 0), 4);
    strongData.lumen         = 8000;
    const LightHandle strong = AddLight(strongData);
    RenderFrame();
    EXPECT_EQ(Stats().staticRenders, 4u);
    EXPECT_EQ(Stats().queue, 8u);
    auto info = [&](LightHandle h)
    {
        for (const FrameLightInfo& f : scene->frameLights())
            if (f.handle == h.id) return f;
        return FrameLightInfo();
    };
    // The strong light first: 4 of its 6 faces, not ready yet (unshadowed); then its last 2 and the weak one's first 2
    EXPECT_FALSE(info(strong).shadowReady);
    EXPECT_EQ(info(strong).tiles, 6);
    RenderFrame();
    EXPECT_TRUE(info(strong).shadowReady);
    EXPECT_FALSE(info(weak).shadowReady);
    RenderFrame();
    EXPECT_TRUE(info(weak).shadowReady);
    EXPECT_EQ(Stats().queue, 0u);
    RenderFrame();
    EXPECT_EQ(Stats().staticRenders, 0u);
}

// A light near the budget's cut keeps its tiles while its importance changes a little (20 % hysteresis)
TEST_F(AtlasTest, HysteresisAvoidsAlternation)
{
    Floor(754);
    // 10 point lights fill 60 of the 64 tiles; an 11th competes with the weakest
    std::vector<LightHandle> h;
    for (int i = 0; i < 10; ++i)
    {
        // On a circle around the camera's foot: every light at the same distance (the importance follows the lumen)
        const float a = 2 * float(kPi) * float(i) / 11;
        LightData d   = ShadowedPoint(vec3(4 * std::cos(a), 2, 4 * std::sin(a)), 4);
        d.lumen       = 1000 + 100.0f * float(i);
        h.push_back(AddLight(d));
    }
    const float a10     = 2 * float(kPi) * 10 / 11;
    LightData rival     = ShadowedPoint(vec3(4 * std::cos(a10), 2, 4 * std::sin(a10)), 4);
    rival.lumen         = 950;
    const LightHandle r = AddLight(rival);
    Capture(2);
    EXPECT_TRUE(scene->shadowAtlas().tilesOf(r.id).empty());
    EXPECT_FALSE(scene->shadowAtlas().tilesOf(h[0].id).empty());
    // Slightly above the weakest (1000): within the hysteresis, nothing changes
    rival.lumen = 1100;
    scene->setLight(r, ChunkId(), rival);
    Capture(2);
    EXPECT_TRUE(scene->shadowAtlas().tilesOf(r.id).empty());
    EXPECT_FALSE(scene->shadowAtlas().tilesOf(h[0].id).empty());
    // Clearly above: the rival takes the weakest's tiles
    rival.lumen = 1500;
    scene->setLight(r, ChunkId(), rival);
    Capture(2);
    EXPECT_FALSE(scene->shadowAtlas().tilesOf(r.id).empty());
    EXPECT_TRUE(scene->shadowAtlas().tilesOf(h[0].id).empty());
}

// The linear depth of D16: a floor at 0.9 range under a point light is lit (no acne), a box on it shadows it
TEST_F(AtlasTest, NoAcneAtRange)
{
    FloorAndBox(756, vec3(0, 0.5f, 0));
    const float range = 10;
    AddLight(ShadowedPoint(vec3(0, 0.9f * range, 0), range));
    const Vec3 eye(0, 12, 0.01);
    Capture(4);
    // Lit floor 2 m beside the box, the box's top and the floor next to the box towards -x (lit: the light is above)
    const vec3 floor = ReadRadiance(Pixel(Vec3(3, 0, 1), eye));
    const vec3 top   = ReadRadiance(Pixel(Vec3(0, 1, 0), eye));
    EXPECT_GT(floor(1), 0.0f);
    EXPECT_GT(top(1), floor(1)) << "the top is nearer";
    // The same frame without the atlas shadow (castShadows off): the lit pixels are as bright (no acne)
    std::vector<LightHandle> keep = lights;
    for (LightHandle l : keep) scene->removeLight(l);
    lights.clear();
    LightData unshadowed   = ShadowedPoint(vec3(0, 0.9f * range, 0), range);
    unshadowed.castShadows = false;
    AddLight(unshadowed);
    Capture(2);
    const vec3 floor2 = ReadRadiance(Pixel(Vec3(3, 0, 1), eye));
    EXPECT_NEAR(floor(1), floor2(1), 0.02f * floor2(1));
}

// A box inside a point light's sphere (the light beside it at its height) shadows the floor behind it, in every
// direction from the light (the six faces)
TEST_F(AtlasTest, PointLightSixFaces)
{
    Floor(758);
    MaterialData m;
    scene->addMaterial(MaterialId(759), "atlas_pillar", m);
    scene->addMesh(MeshId(759), "atlas_pillar", boxMesh(vec3(0.3f, 1.5f, 0.3f), ucvec4(200, 200, 200, 255)),
                   MaterialId(759));
    // Four pillars around the light, at +x, -x, +z, -z
    AddSet({instanceAt(vec3(2, 1.5f, 0), MeshId(759)), instanceAt(vec3(-2, 1.5f, 0), MeshId(759)),
            instanceAt(vec3(0, 1.5f, 2), MeshId(759)), instanceAt(vec3(0, 1.5f, -2), MeshId(759))});
    AddLight(ShadowedPoint(vec3(0, 1.0f, 0), 12));
    const Vec3 eye(0, 12, 0.01);
    Capture(4);
    for (const Vec3& dir : {Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 0, 1), Vec3(0, 0, -1)})
    {
        // Behind the pillar (4 m from the light) and beside the shadow (rotated by 45 degrees)
        const Vec3 behind = dir * 4.0;
        const Vec3 side   = Vec3(dir(0) - dir(2), 0, dir(2) + dir(0)).normalized() * 4.0;
        const vec3 s      = ReadRadiance(Pixel(behind, eye));
        const vec3 l      = ReadRadiance(Pixel(side, eye));
        EXPECT_LT(s(1), 0.2f * l(1)) << dir.transpose();
    }
    EXPECT_EQ(Stats().tilesUsed, 6u);
}
