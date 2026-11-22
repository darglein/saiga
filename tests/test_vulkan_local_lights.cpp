/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The local lights of S36b on RenderScene (docs/stages/S36_lighting.plan.md, B2-B4, Tests): the light selection and
// budget, the falloff and cones, flicker, the per-chunk lists, enclosed spaces; the clusters against a CPU reference
// and the brute-force loop; the cached shadow atlas. One scene per test suite (RenderSceneSuiteTest), skipped without a
// Vulkan 1.3 device.

#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/LocalLights.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include <algorithm>
#include <cmath>
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
