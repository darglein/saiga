/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lighting of S36 on RenderScene (docs/stages/S36_lighting.plan.md, Tests): a1b the LIGHTING_PBR variant (the BRDF
// LUT, energy conservation, the studio environment of clay mode, the prototype colours) and the debug core. One scene
// per test suite (RenderSceneSuiteTest), skipped without a Vulkan 1.3 device.

#include "saiga/vulkan/scene/BrdfLut.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/SkyAmbient.h"

#include <algorithm>
#include <cmath>

#include "render_scene_test_context.h"
#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Vulkan;
using namespace TerrainTestHooks;

namespace
{
constexpr double kPi = 3.14159265358979323846;

// The camera rotation looking along forward (y up)
Quat lookRotation(const Vec3& forward)
{
    const Vec3 back  = -forward.normalized();
    const Vec3 right = Vec3(0, 1, 0).cross(back).normalized();
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

// A UV sphere of radius 1 (smooth normals) in one LOD
MeshLods sphereMesh(const ucvec4& color, int rings = 24, int segments = 48)
{
    MeshLod lod;
    lod.endDistance = 1000;
    for (int r = 0; r <= rings; ++r)
    {
        const float theta = float(kPi) * float(r) / float(rings);
        for (int s = 0; s <= segments; ++s)
        {
            const float phi = 2.0f * float(kPi) * float(s) / float(segments);
            const vec3 n(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));
            MeshVertex v;
            v.position = n;
            v.normal   = packNormalOct(n);
            v.color    = packColor(color);
            lod.vertices.push_back(v);
        }
    }
    const uint32_t w = uint32_t(segments + 1);
    for (uint32_t r = 0; r < uint32_t(rings); ++r)
        for (uint32_t s = 0; s < uint32_t(segments); ++s)
        {
            const uint32_t a = r * w + s, b = a + w;
            for (uint32_t i : {a, a + 1, b, a + 1, b + 1, b}) lod.indices.push_back(i);
        }
    MeshLods m;
    m.lods.push_back(lod);
    m.sphereRadius = 1;
    return m;
}

// A box of the half size, flat faces
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
    lod.endDistance = 1000;
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

double srgbToLinear(double c)
{
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}
double aces(double x)
{
    return std::clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
// The exposed value x of an ACES output y (bisection; ACES is monotonic)
double invertAces(double y)
{
    double lo = 0, hi = 64;
    for (int i = 0; i < 60; ++i)
    {
        const double m          = 0.5 * (lo + hi);
        (aces(m) < y ? lo : hi) = m;
    }
    return 0.5 * (lo + hi);
}

// The mean colour of the pixels within radius of (cx, cy)
vec3 meanColor(const TemplatedImage<ucvec4>& img, int cx, int cy, int radius)
{
    Vec3 sum(0, 0, 0);
    int n = 0;
    for (int y = cy - radius; y <= cy + radius; ++y)
        for (int x = cx - radius; x <= cx + radius; ++x)
        {
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) > radius * radius) continue;
            if (x < 0 || y < 0 || x >= img.w || y >= img.h) continue;
            const ucvec4 p = img(y, x);
            sum += Vec3(p(0), p(1), p(2));
            n++;
        }
    return n ? (sum / double(n) / 255.0).cast<float>() : vec3(0, 0, 0);
}

// An environment without bodies, fog or ambient floor at a fixed EV100
EnvironmentState darkEnvironment(float ev100)
{
    EnvironmentState e;
    e.sunIlluminance  = 0;
    e.moonIlluminance = 0;
    e.fogDensity      = 0;
    e.minAmbient      = vec3(0, 0, 0);
    e.ev100           = ev100;
    return e;
}

class Pbr : public RenderSceneSuiteTest
{
};
}  // namespace

// The LUT of brdf_lut.comp (2048 Hammersley samples, RG16F) matches a CPU integration of the same estimator with 4096
// samples within 0.01 at 8 (NdotV, roughness) texels
TEST_F(Pbr, BrdfLutMatchesCpuIntegration)
{
    const std::vector<vec2> lut = scene->brdfLut().read();
    const int n                 = BrdfLut::kSize;
    ASSERT_EQ(lut.size(), size_t(n) * n);
    const int points[8][2] = {{8, 8}, {32, 8}, {56, 8}, {8, 32}, {32, 32}, {56, 32}, {16, 56}, {48, 56}};
    for (const auto& p : points)
    {
        const float NoV = (float(p[0]) + 0.5f) / float(n), rough = (float(p[1]) + 0.5f) / float(n);
        const vec2 cpu = BrdfLut::integrate(NoV, rough, 4096);
        const vec2 gpu = lut[size_t(p[1]) * size_t(n) + size_t(p[0])];
        EXPECT_NEAR(gpu(0), cpu(0), 0.01) << "NdotV " << NoV << " roughness " << rough;
        EXPECT_NEAR(gpu(1), cpu(1), 0.01) << "NdotV " << NoV << " roughness " << rough;
        // The directional albedo of a single-scattering lobe is at most 1
        EXPECT_LE(gpu(0) + gpu(1), 1.01f);
    }
}

// Energy conservation: a white dielectric and a white metal sphere under a uniform white environment (radiance 1, no
// sun or moon) reflect the environment's radiance within 3% for roughness 0.2, 0.5 and 1 (the multi-scatter
// compensation of Fdez-Agüera); read through the ACES output, inverted
TEST_F(Pbr, WhiteFurnace)
{
    UsePbr();
    scene->setToneMap(ToneMap::Aces);
    // The exposure maps radiance 1 to 0.18
    const float ev100 = float(std::log2(1.0 / (1.2 * 0.18)));
    scene->setEnvironment(darkEnvironment(ev100));
    scene->skyAmbient().setUniformRadiance(vec3(1, 1, 1));
    SetCamera(0, Vec3(0, 0, 4), Quat::Identity(), 0.6);
    const double rPx = height / 2.0 * (1.0 / std::sqrt(15.0)) / std::tan(0.3);
    MeshId mesh      = 100;
    for (float metal : {0.0f, 1.0f})
        for (float rough : {0.2f, 0.5f, 1.0f})
        {
            MaterialData m;
            m.roughness = rough;
            m.metalness = metal;
            scene->addMaterial(MaterialId(mesh), "furnace" + std::to_string(mesh), m);
            scene->addMesh(mesh, "furnace" + std::to_string(mesh), sphereMesh(ucvec4(255, 255, 255, 255)),
                           MaterialId(mesh));
            const auto set = AddSet({instanceAt(vec3(0, 0, 0), mesh)});
            const auto img = Capture(3);
            const vec3 c   = meanColor(img, width / 2, height / 2, int(rPx * 0.8));
            for (int ch = 0; ch < 3; ++ch)
            {
                const double x = invertAces(srgbToLinear(double(c(ch)))) / double(scene->frameExposure());
                EXPECT_NEAR(x, 1.0, 0.03) << "metalness " << metal << " roughness " << rough << " channel " << ch;
            }
            scene->removeInstanceSet(set);
            sets.pop_back();
            mesh++;
        }
}

// Clay mode lights with the studio environment: noon and midnight with clouds render the same image
TEST_F(Pbr, ClayUsesStudioEnvironment)
{
    UsePbr();
    scene->addMaterial(200, "clay_box", MaterialData());
    scene->addMesh(200, "clay_box", boxMesh(vec3(1, 1, 1), ucvec4(200, 200, 200, 255)), 200);
    AddSet({instanceAt(vec3(0, 0, 0), 200)});
    SetCamera(0, Vec3(2.5, 2, 3.5), lookRotation(Vec3(-2.5, -2, -3.5)), 0.8);
    EnvironmentState noon;
    scene->setEnvironment(noon);
    const auto a = Capture(3);
    EnvironmentState night;
    night.sunDir         = vec3(0, -0.7071f, -0.7071f);
    night.moonDir        = vec3(0, 0.7071f, 0.7071f);
    night.cloudiness     = 0.9f;
    night.ev100          = -1;
    night.exposureSource = ExposureSource::Curve;
    scene->setEnvironment(night);
    const auto b = Capture(3);
    ASSERT_EQ(a.w, b.w);
    int differing = 0;
    for (int y = 0; y < a.h; ++y)
        for (int x = 0; x < a.w; ++x)
            if (a(y, x) != b(y, x)) differing++;
    EXPECT_EQ(differing, 0);
    // The studio light: the three visible faces have different shades
    EXPECT_NE(a(height / 2 - 20, width / 2), a(height / 2 + 15, width / 2 - 15));
}

// Clay keeps the material colours of flat-colour instances: a red box stays red
TEST_F(Pbr, ClayKeepsMaterialColours)
{
    UsePbr();
    scene->addMaterial(210, "red_box", MaterialData());
    scene->addMesh(210, "red_box", boxMesh(vec3(1, 1, 1), ucvec4(220, 30, 30, 255)), 210);
    AddSet({instanceAt(vec3(0, 0, 0), 210)});
    SetCamera(0, Vec3(2.5, 2, 3.5), lookRotation(Vec3(-2.5, -2, -3.5)), 0.8);
    const auto img = Capture(3);
    const vec3 c   = meanColor(img, width / 2, height / 2, 6);
    EXPECT_GT(c(0), 2.5f * c(1));
    EXPECT_GT(c(0), 2.5f * c(2));
}

// The prototype colours (D21) only in the physical path: the terrain's mean colour is gray in clay and green-brown in
// ACES with an environment; a 45-degree slope takes the rock colour (red over green), the flat ground the ground colour
// (green over red)
TEST_F(Pbr, PrototypeColoursOnlyInAces)
{
    UsePbr();
    // Flat at 0 m for x < 2048, a 45-degree slope up along +x beyond
    auto hook = [](uint64_t, int64_t x, int64_t) { return int32_t(std::max<int64_t>(0, x - 2048) * 1024); };
    const TerrainLayout layout = testLayout(16, 2);
    scene->setTerrainPyramid(0, testPyramid(hook, layout));
    renderUntilUploaded(*scene);
    // Looking straight down at the slope's foot: the left half flat, the right half the slope
    SetCamera(0, Vec3(2048, 60, 2000), lookRotation(Vec3(0.001, -1, 0)), 0.9);
    auto halves = [&](const TemplatedImage<ucvec4>& img)
    {
        // The camera's up is +x: the slope in the upper half, the flat ground in the lower half
        std::array<vec3, 2> c = {meanColor(img, width / 2, 3 * height / 4, 10),
                                 meanColor(img, width / 2, height / 4, 10)};
        return c;
    };
    const auto clay = halves(Capture(3));
    for (const vec3& c : clay)
    {
        EXPECT_NEAR(c(0), c(1), 2.0f / 255);
        EXPECT_NEAR(c(1), c(2), 2.0f / 255);
    }
    scene->setToneMap(ToneMap::Aces);
    EnvironmentState noon;
    noon.fogDensity = 0;
    scene->setEnvironment(noon);
    const auto aces = halves(Capture(3));
    const vec3 flat = aces[0], slope = aces[1];
    EXPECT_GT(flat(1), flat(0) + 0.02f);
    EXPECT_GT(flat(1), flat(2) + 0.05f);
    EXPECT_GT(slope(0), slope(1));
    EXPECT_GT(slope(0), slope(2) + 0.02f);
}
