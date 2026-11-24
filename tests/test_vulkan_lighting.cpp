/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lighting of S36 on RenderScene (docs/stages/S36_lighting.plan.md, Tests): a1b the PBR shading (the BRDF
// LUT, energy conservation, the studio environment of clay mode, the prototype colours) and the debug core; a1c the
// atmosphere, the environment map and the cloud layer. A scene per test (RenderSceneTest at 128 x 128), skipped
// without a Vulkan 1.3 device.

#include "saiga/vulkan/scene/Atmosphere.h"
#include "saiga/vulkan/scene/BrdfLut.h"
#include "saiga/vulkan/scene/CloudLayer.h"
#include "saiga/vulkan/scene/EnvironmentMap.h"
#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/SkyAmbient.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <set>

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

// probe_lighting's sequence (RenderScene::requestProbe), rendered until it answers
LightingProbe runProbe(RenderScene& scene,
                       const std::function<void(std::function<void(const LightingProbe&)>)>& request)
{
    // The renderer may answer after this function returned (a probe that did not finish in time): the callback owns
    // the answer
    auto answer = std::make_shared<std::optional<LightingProbe>>();
    request([answer](const LightingProbe& p) { *answer = p; });
    for (int i = 0; i < 2 * RenderScene::kProbeFrames && !answer->has_value(); ++i)
        scene.renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    EXPECT_TRUE(answer->has_value());
    return answer->value_or(LightingProbe());
}

// The LUT as the shader samples it: bilinear between texel centres ((i + 0.5) / n), clamped to the edge texels
vec2 sampleLut(const std::vector<vec2>& lut, float NoV, float rough)
{
    const int n = BrdfLut::kSize;
    auto at     = [&](int x, int y)
    { return lut[size_t(std::clamp(y, 0, n - 1)) * size_t(n) + size_t(std::clamp(x, 0, n - 1))]; };
    const float fx = NoV * float(n) - 0.5f, fy = rough * float(n) - 0.5f;
    const int x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
    const float tx = fx - float(x0), ty = fy - float(y0);
    const vec2 a = at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx;
    const vec2 b = at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx;
    return a * (1 - ty) + b * ty;
}

// pbr.glsl's pbrDirect (GGX, height-correlated Smith, Schlick, Lambert, the energy compensation) in closed form: the
// reference of the probe's direct radiance
vec3 pbrDirectReference(const vec3& n, const vec3& v, const vec3& l, const vec3& albedo, float roughness,
                        float metalness, float E, const vec2& lut)
{
    const double pi  = kPi;
    const double NoL = std::clamp(double(n.dot(l)), 0.0, 1.0), NoV = std::max(double(n.dot(v)), 1e-4);
    const vec3 h     = (v + l).normalized();
    const double NoH = std::clamp(double(n.dot(h)), 0.0, 1.0), VoH = std::clamp(double(v.dot(h)), 0.0, 1.0);
    const double a = double(roughness) * roughness, a2 = a * a;
    const double f = (NoH * a2 - NoH) * NoH + 1, D = a2 / (pi * f * f);
    const double V  = 0.5 / (NoL * std::sqrt(NoV * NoV * (1 - a2) + a2) + NoV * std::sqrt(NoL * NoL * (1 - a2) + a2));
    const double fc = std::pow(1 - VoH, 5.0), Ess = double(lut(0) + lut(1));
    vec3 out;
    for (int c = 0; c < 3; ++c)
    {
        const double f0   = 0.04 * (1 - metalness) + double(albedo(c)) * metalness;
        const double F    = f0 + (1 - f0) * fc;
        const double spec = D * V * F * (1 + f0 * (1 / Ess - 1));
        const double diff = (1 - F) * (1 - metalness) * double(albedo(c)) / pi;
        out(c)            = float((diff + spec) * NoL * E);
    }
    return out;
}

// The lighting tests' scenes: 128 x 128 with 4 samples
class LightingSceneTest : public RenderSceneTest
{
   protected:
    ivec2 Size() const override { return ivec2(128, 128); }
};

class Pbr : public LightingSceneTest
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
    // Computed once per process (the later scenes upload it; it cost about 0.1 s per RenderScene on lavapipe)
    EXPECT_EQ(BrdfLut::computations(), 1);
}

// The probe's radiance against pbr.glsl in closed form at the centre of a sphere (NdotV ~ 1):
// - a grey metal of roughness 1 under a uniform environment of radiance 1: the specular part is F0 A + B and the
//   diffuse part the multi-scattering Fms Ems of the LUT's corner texel (NdotV and roughness at the LUT's upper edge).
//   Regression: the LUT was sampled with the default repeat address mode, so at NdotV ~ 1 and roughness 1 the filter
//   mixed in the texels of NdotV ~ 0 and roughness ~ 0 (about half the weight);
// - a grey dielectric of roughness 0.5 in sunlight (no ambient): the direct radiance of GGX, Smith, Schlick and Lambert
//   with the energy compensation (S36a1c: times the atmosphere's transmittance, measured there, so the metal below
//   checks the GGX lobe against it);
// - a white metal of roughness 0.2 at its highlight under 100 000 lux (about 5e6 cd/m^2): finite. Regression: the raw
//   probe frames wrote cd/m^2 without the pre-exposure into the fp16 target (inf above 65504)
TEST_F(Pbr, ProbeMatchesClosedForm)
{
    scene->setToneMap(ToneMap::Aces);
    const std::vector<vec2> lut = scene->brdfLut().read();
    const ivec2 centre(width / 2, height / 2);
    auto sphere = [&](MeshId mesh, float rough, float metal, const ucvec4& color)
    {
        MaterialData m;
        m.roughness = rough;
        m.metalness = metal;
        scene->addMaterial(MaterialId(mesh), "probe_sphere" + std::to_string(mesh), m);
        scene->addMesh(mesh, "probe_sphere" + std::to_string(mesh), sphereMesh(color), MaterialId(mesh));
        return AddSet({instanceAt(vec3(0, 0, 0), mesh)});
    };
    auto probe = [&]() { return runProbe(*scene, [&](auto cb) { scene->requestProbe(centre, cb); }); };

    // The uniform environment
    scene->setEnvironment(darkEnvironment(float(std::log2(1.0 / (1.2 * 0.18)))));
    scene->skyAmbient().setUniformRadiance(vec3(1, 1, 1));
    SetCamera(0, Vec3(0, 0, 4), Quat::Identity(), 0.6);
    auto set = sphere(110, 1.0f, 1.0f, ucvec4(188, 188, 188, 255));
    Capture(2);
    LightingProbe p = probe();
    ASSERT_TRUE(p.hit);
    EXPECT_NEAR(p.roughness, 1.0f, 1e-3f);
    const vec2 corner = lut[size_t(BrdfLut::kSize) * BrdfLut::kSize - 1];
    for (int c = 0; c < 3; ++c)
    {
        const double f0 = p.albedo(c), Ess = corner(0) + corner(1), Ems = 1 - Ess;
        const double FssEss = f0 * corner(0) + corner(1);
        const double Favg   = f0 + (1 - f0) / 21;
        const double Fms    = FssEss * Favg / (1 - Ems * Favg);
        EXPECT_NEAR(p.specular(c), FssEss, 0.02 * FssEss) << "channel " << c;
        EXPECT_NEAR(p.ambient(c), Fms * Ems, 0.02 * FssEss) << "channel " << c;
    }
    scene->removeInstanceSet(set);
    sets.pop_back();

    // The sun alone, from the camera's direction
    const vec3 l         = vec3(0, 0.3f, 1).normalized();
    EnvironmentState sun = darkEnvironment(15);
    sun.sunDir           = l;
    sun.sunIlluminance   = 100000;
    scene->setEnvironment(sun);
    scene->skyAmbient().setUniformRadiance(vec3(0, 0, 0));
    SetCamera(0, (l * 4).cast<double>(), lookRotation(-l.cast<double>()), 0.6);
    vec3 transmittance(1, 1, 1);
    for (int k = 0; k < 2; ++k)
    {
        const float rough = k == 0 ? 0.5f : 0.2f, metal = k == 0 ? 0.0f : 1.0f;
        set = sphere(MeshId(111 + k), rough, metal, k == 0 ? ucvec4(188, 188, 188, 255) : ucvec4(255, 255, 255, 255));
        Capture(2);
        p = probe();
        ASSERT_TRUE(p.hit);
        const vec3 v  = (Vec3(l.cast<double>() * 4) - p.position).normalized().cast<float>();
        const float r = std::clamp(p.roughness, 0.045f, 1.0f);
        const vec3 n  = p.normal.normalized();
        const vec3 expect =
            pbrDirectReference(n, v, l, p.albedo, r, p.metalness, 100000, sampleLut(lut, std::max(n.dot(v), 1e-4f), r));
        for (int c = 0; c < 3; ++c)
        {
            EXPECT_TRUE(std::isfinite(p.direct(c))) << "roughness " << rough;
            // S36a1c: the sun reaches the ground through the atmosphere's transmittance (per channel, < 1): measured
            // on the dielectric, the metal's GGX lobe is checked against it
            if (k == 0)
            {
                transmittance(c) = p.direct(c) / expect(c);
                EXPECT_GT(transmittance(c), 0.3f) << "channel " << c;
                EXPECT_LE(transmittance(c), 1.01f) << "channel " << c;
            }
            else
                EXPECT_NEAR(p.direct(c), transmittance(c) * expect(c), 0.05f * transmittance(c) * expect(c))
                    << "roughness " << rough << " channel " << c;
        }
        if (k == 1)
        {
            EXPECT_GT(p.direct(0), 65504.0f);
        }
        scene->removeInstanceSet(set);
        sets.pop_back();
    }
}

// A normal facing away from the viewer reflects no direct specular light towards it: a front-facing quad whose normals
// face away, a white metal (no diffuse part) of roughness 1 lit from behind along its normal. Regression: the specular
// term was evaluated with the floored NdotV, a highlight on the back of the surface (CodeRabbit #82)
TEST_F(Pbr, NoSpecularFacingAway)
{
    scene->setToneMap(ToneMap::Aces);
    MeshLod lod;
    lod.endDistance = 1000;
    for (const vec3& c : {vec3(-1, -1, 0), vec3(1, -1, 0), vec3(1, 1, 0), vec3(-1, 1, 0)})
    {
        MeshVertex v;
        v.position = c;
        v.normal   = packNormalOct(vec3(0, 0, -1));
        v.color    = packColor(ucvec4(255, 255, 255, 255));
        lod.vertices.push_back(v);
    }
    lod.indices = {0, 1, 2, 0, 2, 3};
    MeshLods quad;
    quad.lods.push_back(lod);
    quad.sphereRadius = std::sqrt(2.0f);
    MaterialData m;
    m.roughness = 1;
    m.metalness = 1;
    scene->addMaterial(MaterialId(115), "facing_away", m);
    scene->addMesh(115, "facing_away", quad, MaterialId(115));
    AddSet({instanceAt(vec3(0, 0, 0), 115)});
    EnvironmentState sun = darkEnvironment(15);
    sun.sunDir           = vec3(0, 0, -1);
    sun.sunIlluminance   = 100000;
    scene->setEnvironment(sun);
    scene->skyAmbient().setUniformRadiance(vec3(0, 0, 0));
    SetCamera(0, Vec3(0, 0, 4), Quat::Identity(), 0.6);
    Capture(2);
    const LightingProbe p = runProbe(*scene, [&](auto cb) { scene->requestProbe(ivec2(width / 2, height / 2), cb); });
    ASSERT_TRUE(p.hit);
    EXPECT_LT(p.normal.z(), -0.99f);
    for (int c = 0; c < 3; ++c) EXPECT_LT(p.direct(c), 1.0f) << "channel " << c;
}

// Energy conservation: a white dielectric and a white metal sphere under a uniform white environment (radiance 1, no
// sun or moon) reflect the environment's radiance within 3% for roughness 0.2, 0.5 and 1 (the multi-scatter
// compensation of Fdez-Agüera); read through the ACES output, inverted
TEST_F(Pbr, WhiteFurnace)
{
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
    noon.cloudiness = 0;
    scene->setEnvironment(noon);
    // The environment map of this environment in the first frame (it updates incrementally otherwise)
    LightingParameters full = scene->lightingParameters();
    full.forceFullUpdates   = true;
    scene->setLightingParameters(full);
    const auto aces = halves(Capture(3));
    const vec3 flat = aces[0], slope = aces[1];
    EXPECT_GT(flat(1), flat(0) + 0.02f);
    EXPECT_GT(flat(1), flat(2) + 0.05f);
    EXPECT_GT(slope(0), slope(1));
    EXPECT_GT(slope(0), slope(2) + 0.02f);
}

// ---------------------------------------------------------------------------------------------------------------------
// The debug core (S36a1b): views, overlays, previews, readbacks, the probe

namespace
{
class LightingDebugTest : public LightingSceneTest
{
   protected:
    // A gray plane (top at y = 0) and a red box on it, the camera 6 m above looking down at the plane
    void PlaneScene(MeshId first)
    {
        MaterialData m;
        m.roughness = 0.6f;
        scene->addMaterial(MaterialId(first), "debug_plane" + std::to_string(first), m);
        scene->addMesh(first, "debug_plane" + std::to_string(first),
                       boxMesh(vec3(20, 0.05f, 20), ucvec4(180, 180, 180, 255)), MaterialId(first));
        scene->addMesh(MeshId(first + 1), "debug_box" + std::to_string(first),
                       boxMesh(vec3(0.5f, 0.5f, 0.5f), ucvec4(220, 40, 40, 255)), MaterialId(first));
        AddSet({instanceAt(vec3(0, -0.05f, 0), first), instanceAt(vec3(2, 0.5f, 0), MeshId(first + 1))});
        SetCamera(0, Vec3(0, 6, 0.01), lookRotation(Vec3(0, -1, -0.002)), 0.9);
    }
    // The resolved fp16 value of a pixel in the next frame
    vec4 ReadPixel(ivec2 p)
    {
        vec4 value(-1, -1, -1, -1);
        scene->lightingDebugImages().requestPixels({p},
                                                   [&](const std::vector<LightingDebug::PixelValue>& v)
                                                   {
                                                       if (!v.empty() && v[0].valid) value = v[0].color;
                                                   });
        RenderFrame();
        return value;
    }
};
}  // namespace

// Every view and the overlays render 2 frames without a validation error (the PBR shading, clay and ACES with
// an environment); the Normals of a plane facing +y read (0.5, 1, 0.5) from the resolved fp16 image (raw data)
TEST_F(LightingDebugTest, EveryViewRenders)
{
    PlaneScene(300);
    for (ToneMap tone : {ToneMap::Identity, ToneMap::Aces})
    {
        scene->setToneMap(tone);
        for (int v = 0; v < int(LightingDebugView::Count); ++v)
        {
            scene->setLightingDebug(LightingDebugState{LightingDebugView(v), OverlayAll});
            Capture(2);
            EXPECT_EQ(scene->frameLightingDebug().view, LightingDebugView(v));
            // A debug frame latches Identity
            EXPECT_EQ(scene->toneMap(), v == 0 ? tone : ToneMap::Identity);
        }
        scene->setLightingDebug(LightingDebugState{LightingDebugView::Normals, 0});
        Capture(1);
        const vec4 n = ReadPixel(ivec2(width / 2 - 20, height / 2));
        EXPECT_NEAR(n(0), 0.5f, 0.01f) << n.transpose();
        EXPECT_NEAR(n(1), 1.0f, 0.01f) << n.transpose();
        EXPECT_NEAR(n(2), 0.5f, 0.01f) << n.transpose();
    }
    scene->setLightingDebug(LightingDebugState());
}

// Finding 24 of the plan's review 2: no pipeline depends on the tone map or the debug view (the output mode is a push
// constant, the resolve mode is chosen per frame): a view switch and a tone map switch create no pipeline
TEST_F(LightingDebugTest, ViewSwitchCreatesNoPipeline)
{
    PlaneScene(320);
    Capture(2);
    const uint64_t before = scene->scenePipelineCreations();
    for (LightingDebugView v : {LightingDebugView::Albedo, LightingDebugView::Exposure, LightingDebugView::None})
    {
        scene->setLightingDebug(LightingDebugState{v, OverlaySunDirection});
        scene->setToneMap(ToneMap::Aces);
        Capture(2);
        scene->setToneMap(ToneMap::Identity);
        Capture(2);
    }
    EXPECT_EQ(scene->scenePipelineCreations(), before);
    // The counter counts (the scene's pipelines at its creation)
    EXPECT_GT(before, 0u);
}

// A debug frame (Identity latched) does not update the pre-exposure history: the frame after it has the pre-exposure
// of the frame before it (with an effective exposure within kMaxPreExposureChange of 1, so a reset to 1 would show)
TEST_F(LightingDebugTest, DebugFrameKeepsPreExposure)
{
    PlaneScene(340);
    scene->setToneMap(ToneMap::Aces);
    EnvironmentState e;
    e.ev100 = float(std::log2(1.0 / (1.2 * 0.25)));  // effective exposure 0.25
    scene->setEnvironment(e);
    Capture(3);
    const float before = scene->preExposure();
    EXPECT_NEAR(before, 0.25f, 1e-4f);
    scene->setLightingDebug(LightingDebugState{LightingDebugView::Albedo, 0});
    Capture(1);
    EXPECT_EQ(scene->preExposure(), 1.0f);
    scene->setLightingDebug(LightingDebugState());
    Capture(1);
    EXPECT_EQ(scene->preExposure(), before);
}

// The previews: the luminance preview has the frame's size and content, its reduction's min / max match the read-back
// frame (clay: the display luminance, the post pass shows it unchanged); the BRDF LUT's match the LUT read back
TEST_F(LightingDebugTest, PreviewsCompute)
{
    PlaneScene(360);
    Capture(2);
    LightingDebug::Capture lum, lut;
    bool gotLum = false, gotLut = false;
    PreviewMapping linear;
    scene->lightingDebugImages().request(LightingImage::Luminance, linear,
                                         [&](const LightingDebug::Capture& c)
                                         {
                                             lum    = c;
                                             gotLum = true;
                                         });
    scene->lightingDebugImages().request(LightingImage::BrdfLut, linear,
                                         [&](const LightingDebug::Capture& c)
                                         {
                                             lut    = c;
                                             gotLut = true;
                                         });
    const auto frame = RenderFrame();
    ASSERT_TRUE(gotLum && gotLut);
    EXPECT_EQ(scene->lightingStatistics().previewDispatches, 2u);
    ASSERT_EQ(lum.size, ivec2(width, height));
    ASSERT_EQ(lum.image.w, width);
    // Non-constant content (the plane and the box)
    std::set<int> values;
    for (int y = 0; y < lum.image.h; y += 4)
        for (int x = 0; x < lum.image.w; x += 4) values.insert(lum.image(y, x)(0));
    EXPECT_GT(values.size(), 3u);
    // The reduction against the read-back frame's luminance (8-bit)
    float lo = 1e9f, hi = -1e9f;
    for (int y = 0; y < frame.h; ++y)
        for (int x = 0; x < frame.w; ++x)
        {
            const ucvec4 p = frame(y, x);
            const float l  = (0.2126f * p(0) + 0.7152f * p(1) + 0.0722f * p(2)) / 255.0f;
            lo             = std::min(lo, l);
            hi             = std::max(hi, l);
        }
    EXPECT_NEAR(lum.statistics.min, lo, 0.01f);
    EXPECT_NEAR(lum.statistics.max, hi, 0.01f);
    EXPECT_EQ(lum.statistics.count, uint32_t(width * height));
    uint32_t total = 0;
    for (uint32_t h : lum.statistics.histogram) total += h;
    EXPECT_EQ(total, uint32_t(width * height));
    // The BRDF LUT
    ASSERT_EQ(lut.size, ivec2(BrdfLut::kSize, BrdfLut::kSize));
    const std::vector<vec2> cpu = scene->brdfLut().read();
    float a0 = 1e9f, a1 = -1e9f;
    for (const vec2& v : cpu)
    {
        a0 = std::min(a0, v(0));
        a1 = std::max(a1, v(0));
    }
    EXPECT_NEAR(lut.statistics.min, a0, 1e-4f);
    EXPECT_NEAR(lut.statistics.max, a1, 1e-4f);
}

// A preview is computed only in a frame that requested it
TEST_F(LightingDebugTest, NothingComputedWithoutRequest)
{
    PlaneScene(380);
    Capture(2);
    EXPECT_EQ(scene->lightingStatistics().previewDispatches, 0u);
    scene->lightingDebugImages().request(LightingImage::Luminance, PreviewMapping());
    Capture(1);
    EXPECT_EQ(scene->lightingStatistics().previewDispatches, 1u);
    Capture(1);
    EXPECT_EQ(scene->lightingStatistics().previewDispatches, 0u);
    // Its statistics arrive with the slot's next use (one frame slot later)
    Capture(2);
    EXPECT_GT(scene->lightingStatistics().luminanceFrame, 0u);
}

// A shown preview (the Lighting tab, show()) is converted at preview resolution, at most once per preview interval
// unless its mapping changes, and within the per-frame texel budget. Regression: the tab converted the full-resolution
// luminance every frame with a per-pixel histogram atomic on a host-visible buffer (about 150 ms per 1080p frame on an
// RTX 5080 Laptop GPU)
TEST_F(LightingDebugTest, ShownPreviewsAreRateLimited)
{
    PlaneScene(390);
    LightingDebug& d = scene->lightingDebugImages();
    d.setPreviewInterval(1e9);
    d.setPreviewLimits(32, LightingDebug::kPreviewTexelBudget);
    PreviewMapping mapping;
    mapping.log    = true;
    uint32_t total = 0;
    for (int f = 0; f < 4; ++f)
    {
        d.show(LightingImage::Luminance, mapping);
        RenderFrame();
        total += scene->lightingStatistics().previewDispatches;
    }
    EXPECT_EQ(total, 1u);
    // At preview resolution (every step-th texel), with the statistics of the preview's texels
    const int step = (std::max(width, height) + 31) / 32;
    ASSERT_NE(d.preview(LightingImage::Luminance), nullptr);
    EXPECT_EQ(int(d.preview(LightingImage::Luminance)->width), (width + step - 1) / step);
    EXPECT_EQ(int(d.preview(LightingImage::Luminance)->height), (height + step - 1) / step);
    EXPECT_EQ(d.statistics(LightingImage::Luminance).count,
              uint32_t(((width + step - 1) / step) * ((height + step - 1) / step)));
    // A mapping change converts at once; a request() (a capture) always converts, at full resolution
    mapping.colorMap = ColorMap::Turbo;
    d.show(LightingImage::Luminance, mapping);
    RenderFrame();
    EXPECT_EQ(scene->lightingStatistics().previewDispatches, 1u);
    LightingDebug::Capture c;
    d.request(LightingImage::Luminance, mapping, [&](const LightingDebug::Capture& r) { c = r; });
    d.show(LightingImage::Luminance, mapping);
    RenderFrame();
    EXPECT_EQ(scene->lightingStatistics().previewDispatches, 1u);
    EXPECT_EQ(c.size, ivec2(width, height));
    // The texel budget: of two shown images only the first is converted in a frame, the other in the next one
    d.setPreviewInterval(0);
    d.setPreviewLimits(32, 1);
    // Both are converted (their statistics advance; a preview of an earlier test on the scene would pass a null check)
    Capture(2);
    const uint64_t luminanceBefore = d.statisticsFrame(LightingImage::Luminance);
    const uint64_t lutBefore       = d.statisticsFrame(LightingImage::BrdfLut);
    for (int f = 0; f < 2; ++f)
    {
        d.show(LightingImage::Luminance, mapping);
        d.show(LightingImage::BrdfLut, mapping);
        RenderFrame();
        EXPECT_EQ(scene->lightingStatistics().previewDispatches, 1u);
    }
    EXPECT_NE(d.preview(LightingImage::BrdfLut), nullptr);
    // The statistics arrive with the slots' next use
    Capture(2);
    EXPECT_GT(d.statisticsFrame(LightingImage::Luminance), luminanceBefore);
    EXPECT_GT(d.statisticsFrame(LightingImage::BrdfLut), lutBefore);
    d.setPreviewInterval(0.25);
    d.setPreviewLimits(LightingDebug::kPreviewMaxSize, LightingDebug::kPreviewTexelBudget);
}

// Two captures of one image with different mappings in one frame each get their own mapping's preview and statistics
// (the second a frame later), and the tab's show() of that frame changes neither. Regression: the last request's (or
// the tab's) mapping replaced the first one's
TEST_F(LightingDebugTest, RequestsKeepTheirMapping)
{
    PlaneScene(420);
    Capture(2);
    LightingDebug& d = scene->lightingDebugImages();
    PreviewMapping linear, log2Mapping, tab;
    log2Mapping.log = true;
    tab.log         = true;
    tab.colorMap    = ColorMap::Turbo;
    // The callbacks own the captures (a failed assertion returns with a request still queued in the scene)
    auto a = std::make_shared<std::optional<LightingDebug::Capture>>();
    auto b = std::make_shared<std::optional<LightingDebug::Capture>>();
    d.request(LightingImage::Luminance, linear, [a](const LightingDebug::Capture& c) { *a = c; });
    d.request(LightingImage::Luminance, log2Mapping, [b](const LightingDebug::Capture& c) { *b = c; });
    d.show(LightingImage::Luminance, tab);
    for (int i = 0; i < 4 && !(*a && *b); ++i) RenderFrame();
    ASSERT_TRUE(*a && *b);
    EXPECT_GT((*a)->statistics.max, 0.05f);
    EXPECT_NEAR((*b)->statistics.max, std::log2((*a)->statistics.max), 0.01f);
    // Gray, not the tab's turbo
    const ucvec4 c = (*a)->image(height / 2, width / 2 - 20);
    EXPECT_EQ(c(0), c(1));
    EXPECT_EQ(c(1), c(2));
}

// While a probe runs, a screenshot and a capture requested with it and a capture requested during it wait for the frame
// after the sequence, so they read a normal frame. Regression: they read the probe's debug frames (an Albedo / Normals
// / raw radiance image); a capture requested with the probe was converted in its frame 1 (CodeRabbit #82)
TEST_F(LightingDebugTest, ProbeDefersScreenshotsAndCaptures)
{
    PlaneScene(430);
    scene->setToneMap(ToneMap::Aces);
    EnvironmentState noon;
    noon.fogDensity = 0;
    noon.ev100      = 14;
    scene->setEnvironment(noon);
    Capture(3);
    LightingDebug& d = scene->lightingDebugImages();
    PreviewMapping linear;
    LightingDebug::Capture normal, during;
    d.request(LightingImage::Luminance, linear, [&](const LightingDebug::Capture& c) { normal = c; });
    const auto reference = RenderFrame();
    bool probed = false, captured = false, capturedWith = false;
    uint64_t probeFrame = 0, shotFrame = 0, captureFrame = 0, captureWithFrame = 0;
    TemplatedImage<ucvec4> shot;
    scene->requestProbe(ivec2(width / 2 - 20, height / 2),
                        [&](const LightingProbe&)
                        {
                            probed     = true;
                            probeFrame = scene->frames();
                        });
    scene->requestScreenshot(
        [&](const TemplatedImage<ucvec4>& img)
        {
            shot      = img;
            shotFrame = scene->frames();
        });
    d.request(LightingImage::Luminance, linear,
              [&](const LightingDebug::Capture&)
              {
                  capturedWith     = true;
                  captureWithFrame = scene->frames();
              });
    // The capture is requested in the sequence (frame 1 is a normal frame, the later ones are not)
    scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    ASSERT_TRUE(scene->probeRunning());
    d.request(LightingImage::Luminance, linear,
              [&](const LightingDebug::Capture& c)
              {
                  during       = c;
                  captured     = true;
                  captureFrame = scene->frames();
              });
    for (int i = 0; i < 2 * RenderScene::kProbeFrames && !(probed && captured && capturedWith && shot.w > 0); ++i)
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    ASSERT_TRUE(probed && captured && capturedWith && shot.w > 0);
    EXPECT_GT(shotFrame, probeFrame);
    EXPECT_GT(captureFrame, probeFrame);
    EXPECT_GT(captureWithFrame, probeFrame);
    ASSERT_EQ(shot.w, reference.w);
    int maxDiff = 0;
    for (int y = 0; y < shot.h; ++y)
        for (int x = 0; x < shot.w; ++x)
            for (int c = 0; c < 3; ++c)
                maxDiff = std::max(maxDiff, std::abs(int(shot(y, x)(c)) - int(reference(y, x)(c))));
    EXPECT_LE(maxDiff, 1);
    EXPECT_NEAR(during.statistics.mean, normal.statistics.mean, 1e-3f * std::abs(normal.statistics.mean));
}

// A probe at a world point projects it with the camera of the sequence's frame 1 (a camera change before that frame
// is used). Regression: the tool projected with the last frame's matrices around the app's current camera position. A
// point behind the camera is not in the image
TEST_F(LightingDebugTest, ProbeAtWorldPoint)
{
    PlaneScene(440);
    Capture(1);
    const Vec3 world(1.5, 0, -1);
    LightingProbe p = runProbe(*scene,
                               [&](auto cb)
                               {
                                   scene->requestProbe(world, cb);
                                   // The camera moves before the sequence's first frame
                                   SetCamera(0, Vec3(-1, 8, 1), lookRotation(Vec3(0, -1, -0.002)), 0.9);
                               });
    ASSERT_TRUE(p.inImage);
    ASSERT_TRUE(p.hit);
    EXPECT_NEAR(p.position.x(), world.x(), 0.06);
    EXPECT_NEAR(p.position.y(), world.y(), 0.02);
    EXPECT_NEAR(p.position.z(), world.z(), 0.06);
    p = runProbe(*scene, [&](auto cb) { scene->requestProbe(Vec3(0, 20, 0), cb); });
    EXPECT_FALSE(p.inImage);
    // Far off screen (beyond the int range in pixels: the range is checked before the conversion; CodeRabbit #82)
    p = runProbe(*scene, [&](auto cb) { scene->requestProbe(Vec3(1e9, 0, -1), cb); });
    EXPECT_FALSE(p.inImage);
    // The transfer callback of frame 1 sets the camera (the render path's camera): frame 1 uses it, the later frames
    // hold it. Regression: it was held until after the sequence, frame 1 projected with the previous camera, which
    // looks away from the point (CodeRabbit #82)
    SetCamera(0, Vec3(-1, 8, 1), lookRotation(Vec3(0, 1, 0.002)), 0.9);
    Capture(1);
    auto answer = std::make_shared<std::optional<LightingProbe>>();
    scene->requestProbe(world, [answer](const LightingProbe& r) { *answer = r; });
    const auto transfer = [&](vk::CommandBuffer)
    { SetCamera(0, Vec3(-1, 8, 1), lookRotation(Vec3(0, -1, -0.002)), 0.9); };
    for (int i = 0; i < 2 * RenderScene::kProbeFrames && !answer->has_value(); ++i)
        scene->renderFrame(transfer, [](vk::CommandBuffer) {}, nullptr);
    ASSERT_TRUE(answer->has_value());
    p = **answer;
    ASSERT_TRUE(p.inImage);
    ASSERT_TRUE(p.hit);
    EXPECT_TRUE(p.consistent);
    EXPECT_NEAR(p.position.x(), world.x(), 0.06);
    EXPECT_NEAR(p.position.z(), world.z(), 0.06);
}

// All frames of a probe use frame 1's light model: a tone map set during the sequence (Identity: the studio lighting)
// takes effect after it. Regression: the light model followed the tone map per frame, the raw frames 5-7 used the
// studio lighting while the probe reported physical (CodeRabbit #82)
TEST_F(LightingDebugTest, ProbeHoldsToneMap)
{
    PlaneScene(450);
    scene->setToneMap(ToneMap::Aces);
    EnvironmentState noon;
    noon.fogDensity = 0;
    noon.ev100      = 14;
    scene->setEnvironment(noon);
    Capture(2);
    const ivec2 pixel(width / 2 - 20, height / 2);
    const LightingProbe reference = runProbe(*scene, [&](auto cb) { scene->requestProbe(pixel, cb); });
    ASSERT_TRUE(reference.hit);
    auto answer = std::make_shared<std::optional<LightingProbe>>();
    scene->requestProbe(pixel, [answer](const LightingProbe& r) { *answer = r; });
    // (no screenshots: they wait for the frame after the sequence)
    const auto frame = [&]() { scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr); };
    frame();
    ASSERT_TRUE(scene->probeRunning());
    scene->setToneMap(ToneMap::Identity);
    for (int i = 0; i < 2 * RenderScene::kProbeFrames && !answer->has_value(); ++i) frame();
    ASSERT_TRUE(answer->has_value());
    const LightingProbe& p = **answer;
    EXPECT_TRUE(p.physical);
    for (int c = 0; c < 3; ++c)
    {
        EXPECT_NEAR(p.direct(c), reference.direct(c), 0.01f * reference.direct(c)) << "channel " << c;
        EXPECT_NEAR(p.ambient(c), reference.ambient(c), 0.01f * reference.ambient(c)) << "channel " << c;
    }
    // Applied after the sequence
    EXPECT_EQ(scene->parameters().toneMap, ToneMap::Identity);
    scene->setToneMap(ToneMap::Aces);
}

// The probe's colour is the radiance without the pre-exposure also with ACES and no environment (linear output, not
// physical): independent of the exposure multiplier. Regression (CodeRabbit #82, fixed in a1b; the test needs a scene
// without an environment, which the a1b suite fixture did not have): only a physical frame was divided, so the colour
// grew with the exposure
TEST_F(LightingDebugTest, ProbeColourWithoutEnvironment)
{
    PlaneScene(450);
    scene->setToneMap(ToneMap::Aces);
    const ivec2 pixel(width / 2 - 20, height / 2);
    auto probeAt = [&](float exposure)
    {
        scene->setExposure(exposure);
        Capture(3);
        return runProbe(*scene, [&](auto cb) { scene->requestProbe(pixel, cb); });
    };
    const LightingProbe a = probeAt(1);
    const LightingProbe b = probeAt(4);
    ASSERT_TRUE(a.hit && b.hit);
    EXPECT_FALSE(a.physical);
    EXPECT_GT(a.luminance, 0);
    EXPECT_NEAR(b.luminance, a.luminance, 0.01f * a.luminance);
}

// probe_lighting's sequence: 7 frames at a pixel of the lit plane, the values equal the views' at that pixel, the
// camera and the environment are held (a setEnvironment during it takes effect after it), consistent
TEST_F(LightingDebugTest, ProbeMatchesViews)
{
    PlaneScene(400);
    scene->setToneMap(ToneMap::Aces);
    EnvironmentState noon;
    noon.fogDensity = 0;
    noon.ev100      = 14;
    scene->setEnvironment(noon);
    Capture(2);
    const ivec2 pixel(width / 2 - 20, height / 2);
    // The views' values at the pixel
    scene->setLightingDebug(LightingDebugState{LightingDebugView::Albedo, 0});
    Capture(1);
    const vec4 albedo = ReadPixel(pixel);
    scene->setLightingDebug(LightingDebugState{LightingDebugView::Material, 0});
    Capture(1);
    const vec4 material = ReadPixel(pixel);
    scene->setLightingDebug(LightingDebugState());
    Capture(1);

    bool answered = false;
    LightingProbe probe;
    const uint64_t startFrame = scene->frames();
    scene->requestProbe(pixel,
                        [&](const LightingProbe& p)
                        {
                            probe    = p;
                            answered = true;
                        });
    RenderFrame();
    EXPECT_TRUE(scene->probeRunning());
    // Held during the sequence: the environment of frame 1
    EnvironmentState later = noon;
    later.ev100            = 9;
    scene->setEnvironment(later);
    EXPECT_EQ(scene->environment().ev100, 14.0f);
    for (int i = 0; i < 10 && !answered; ++i) RenderFrame();
    ASSERT_TRUE(answered);
    EXPECT_EQ(scene->frames() - startFrame, uint64_t(RenderScene::kProbeFrames));
    EXPECT_EQ(scene->environment().ev100, 9.0f);
    EXPECT_EQ(probe.frames, RenderScene::kProbeFrames);
    EXPECT_TRUE(probe.inImage);
    EXPECT_TRUE(probe.hit);
    EXPECT_TRUE(probe.consistent);
    EXPECT_TRUE(probe.physical);
    for (int c = 0; c < 3; ++c) EXPECT_NEAR(probe.albedo(c), albedo(c), 1e-3f);
    EXPECT_NEAR(probe.roughness, material(0), 1e-3f);
    EXPECT_NEAR(probe.metalness, material(1), 1e-3f);
    EXPECT_NEAR(probe.normal(1), 1.0f, 0.01f);
    EXPECT_GT(probe.direct(0), 0.0f);
    EXPECT_GT(probe.ambient(0), 0.0f);
    EXPECT_GT(probe.luminance, 0.0f);
    // The plane at y = 0, 6 m below the camera
    EXPECT_NEAR(probe.position.y(), 0.0, 0.05);
    // The user's view is unchanged
    EXPECT_EQ(scene->lightingDebug().view, LightingDebugView::None);
}

// ---------------------------------------------------------------------------------------------------------------------
// The atmosphere and the environment map (S36a1c)

namespace
{
// A texel of an RGBA16F download
vec3 halfTexel(const std::vector<uint8_t>& bytes, int x, int y, int w)
{
    uint16_t h[4];
    std::memcpy(h, bytes.data() + (size_t(y) * size_t(w) + size_t(x)) * 8, 8);
    return vec3(LightingDebug::halfToFloat(h[0]), LightingDebug::halfToFloat(h[1]), LightingDebug::halfToFloat(h[2]));
}

// The CPU reference of the transmittance (Hillaire's Earth medium, a dense midpoint integration in double) from the
// radius r (km) along the cosine mu to the top of the atmosphere
vec3 referenceTransmittance(double r, double mu)
{
    const double G = 6360, T = 6460;
    const double d = -r * mu + std::sqrt(std::max(r * r * (mu * mu - 1) + T * T, 0.0));
    const int n    = 20000;
    double opt[3]  = {0, 0, 0};
    for (int i = 0; i < n; ++i)
    {
        const double t  = d * (i + 0.5) / n;
        const double h  = std::sqrt(r * r + t * t + 2 * r * mu * t) - G;
        const double rd = std::exp(-h / 8), md = std::exp(-h / 1.2), od = std::max(0.0, 1 - std::abs(h - 25) / 15);
        const double ray[3] = {5.802e-3, 13.558e-3, 33.1e-3}, oz[3] = {0.650e-3, 1.881e-3, 0.085e-3};
        for (int c = 0; c < 3; ++c) opt[c] += (ray[c] * rd + 4.40e-3 * md + oz[c] * od) * d / n;
    }
    return vec3(float(std::exp(-opt[0])), float(std::exp(-opt[1])), float(std::exp(-opt[2])));
}

// The (r, mu) of a transmittance LUT texel centre (atmosphere_common.glsl's transmittanceRMu)
void transmittanceRMu(double u, double v, double& r, double& mu)
{
    const double G = 6360, T = 6460, H = std::sqrt(T * T - G * G);
    const double rho = H * v;
    r                = std::sqrt(rho * rho + G * G);
    const double dm = T - r, dM = rho + H, d = dm + u * (dM - dm);
    mu = d == 0 ? 1.0 : std::clamp((H * H - rho * rho - d * d) / (2 * r * d), -1.0, 1.0);
}

EnvironmentState sunAt(float elevationDeg, float azimuthDeg = 180)
{
    // Azimuth clockwise from north (-z): 180 is south (+z)
    EnvironmentState e;
    const float el = elevationDeg * float(kPi) / 180, az = azimuthDeg * float(kPi) / 180;
    e.sunDir     = vec3(std::cos(el) * std::sin(az), std::sin(el), -std::cos(el) * std::cos(az));
    e.moonDir    = -e.sunDir;
    e.fogDensity = 0;
    return e;
}

class AtmosphereTest : public LightingSceneTest
{
   protected:
    // The physical path (ACES) with the environment, the camera 100 m up looking along forward; the environment map
    // is updated fully in the next frame (a settled state in one frame; test speed), everyFrame: in every frame
    // (forceFullUpdates)
    void Physical(const EnvironmentState& e, const Vec3& forward = Vec3(0, 0.3, 1), double fovy = 1.2,
                  bool everyFrame = false)
    {
        LightingParameters p    = scene->lightingParameters();
        p.forceFullUpdates      = everyFrame;
        p.environmentResolution = CubeResolution();
        scene->setLightingParameters(p);
        scene->requestFullUpdate();
        scene->setToneMap(ToneMap::Aces);
        scene->setEnvironment(e);
        SetCamera(0, Vec3(0, 100, 0), lookRotation(forward), fovy);
    }
    float luminance(const ucvec4& p) const { return 0.2126f * p(0) + 0.7152f * p(1) + 0.0722f * p(2); }
    // The environment map's resolution: the smallest where a suite does not check the cube's content
    virtual int CubeResolution() const { return 32; }
};
}  // namespace

// The transmittance LUT against a dense CPU integration of the same medium at the texel centres (zenith to horizon)
TEST_F(AtmosphereTest, TransmittanceReference)
{
    Physical(sunAt(45));
    Capture(2);
    std::vector<uint8_t> bytes = scene->atmosphereLuts().transmittance().download();
    const int w = int(Atmosphere::kTransmittanceWidth), h = int(Atmosphere::kTransmittanceHeight);
    ASSERT_EQ(bytes.size(), size_t(w) * h * 8);
    for (int y : {0, 10, 40})
        for (int x : {0, 64, 128, 200, 255})
        {
            double r, mu;
            transmittanceRMu((x + 0.5) / w, (y + 0.5) / h, r, mu);
            const vec3 ref = referenceTransmittance(r, mu);
            const vec3 gpu = halfTexel(bytes, x, y, w);
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(gpu(c), ref(c), 0.02f * std::max(ref(c), 0.05f) + 2e-3f)
                    << "texel " << x << ", " << y << " channel " << c;
        }
    // The zenith at the sea level (Earth): about 0.94 red, 0.76 blue
    double r, mu;
    transmittanceRMu(0.5 / w, 0.5 / h, r, mu);
    const vec3 zenith = halfTexel(bytes, 0, 0, w);
    EXPECT_GT(zenith(0), zenith(2));
    EXPECT_NEAR(zenith(0), 0.94f, 0.02f);
}

namespace
{
// The sky-view LUT's texel of a direction (elevation in degrees, azimuth atan(z, x) in degrees) for the camera 100 m
// above the sea level (AtmosphereTest::Physical), through the CPU copy of the mapping
vec3 skyViewTexel(const std::vector<uint8_t>& lut, float elevationDeg, float azimuthDeg)
{
    const int w = int(Atmosphere::kSkyViewWidth), h = int(Atmosphere::kSkyViewHeight);
    const float el = elevationDeg * float(kPi) / 180, az = azimuthDeg * float(kPi) / 180;
    const vec2 uv = Atmosphere::skyViewUv(vec3(std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)),
                                          Atmosphere::skyViewHorizon(Atmosphere::radiusOf(100)));
    return halfTexel(lut, std::clamp(int(uv(0) * float(w)), 0, w - 1), std::clamp(int(uv(1) * float(h)), 0, h - 1), w);
}
}  // namespace

// The sky-view LUT's mapping (CPU copy of atmosphere_common.glsl): the horizon is the planet's limb (-0.32 degrees at
// 100 m), uv and direction round-trip, and the rows concentrate at the horizon. Regression: the horizon was computed
// as -acos(sqrt(r^2 - Rg^2) / r) = -89.7 degrees (the angle to the nadir), which put half of the rows next to the
// nadir and about 4.8 degrees of elevation into each row at the horizon
TEST(SkyViewMapping, HorizonAndRows)
{
    const float horizon = Atmosphere::skyViewHorizon(Atmosphere::radiusOf(100));
    const float pif     = 3.14159265f;
    const float degrees = horizon * 180 / pif;
    EXPECT_NEAR(degrees, -0.32f, 0.01f);
    const float v0 = Atmosphere::skyViewUv(vec3(1, 0, 0), horizon)(1);
    EXPECT_NEAR(v0, 0.5f + 0.5f * std::sqrt(-horizon / (0.5f * pif - horizon)), 1e-6f);
    for (float v : {0.02f, 0.3f, 0.49f, 0.51f, 0.6f, 0.97f})
        for (float u : {0.1f, 0.5f, 0.8f})
        {
            const vec2 back = Atmosphere::skyViewUv(Atmosphere::skyViewDir(vec2(u, v), horizon), horizon);
            EXPECT_NEAR(back(0), u, 1e-4f) << u << ", " << v;
            EXPECT_NEAR(back(1), v, 1e-4f) << u << ", " << v;
        }
    // The rows whose centre is within 5 degrees of the elevation 0
    int rows = 0;
    for (int y = 0; y < int(Atmosphere::kSkyViewHeight); ++y)
    {
        const vec3 d =
            Atmosphere::skyViewDir(vec2(0.5f, (float(y) + 0.5f) / float(Atmosphere::kSkyViewHeight)), horizon);
        if (std::abs(std::asin(d(1))) < 5 * pif / 180) rows++;
    }
    EXPECT_GE(rows, 15);
}

// The sky-view LUT: blue at the zenith at noon, sky above and ground below the horizon row, red near the horizon
// towards the sun at sunset (sampled by elevation through the CPU copy of the mapping: the old test read fixed rows
// that were ground texels under the wrong horizon)
TEST_F(AtmosphereTest, SkyIsBlueAtNoonAndRedAtSunset)
{
    const int w = int(Atmosphere::kSkyViewWidth), h = int(Atmosphere::kSkyViewHeight);
    Physical(sunAt(60));
    Capture(2);
    std::vector<uint8_t> noon = scene->atmosphereLuts().skyView().download();
    const vec3 zenith         = halfTexel(noon, w / 2, h - 1, w);
    EXPECT_GT(zenith(2), 1.5f * zenith(0)) << zenith.transpose();
    // The row just above v = 0.5 is the sky at the horizon (bright, bluish), the row just below the ground (darker)
    const vec3 skyRow = halfTexel(noon, w / 4, h / 2, w), groundRow = halfTexel(noon, w / 4, h / 2 - 1, w);
    EXPECT_GT(skyRow(2), skyRow(0)) << skyRow.transpose();
    EXPECT_GT(skyRow.sum(), 1.2f * groundRow.sum()) << skyRow.transpose() << " / " << groundRow.transpose();
    // A few degrees up the sky is blue, a few degrees down the ground is darker than it
    for (float el : {1.0f, 3.0f, 10.0f})
    {
        const vec3 sky = skyViewTexel(noon, el, 0), ground = skyViewTexel(noon, -el - 0.5f, 0);
        EXPECT_GT(sky(2), sky(0)) << el << ": " << sky.transpose();
        EXPECT_GT(sky.sum(), ground.sum()) << el << ": " << sky.transpose() << " / " << ground.transpose();
    }
    // Sunset in the south (+z): the azimuth atan(z, x) = 90 degrees; two degrees above the horizon
    Physical(sunAt(1));
    Capture(2);
    std::vector<uint8_t> sunset = scene->atmosphereLuts().skyView().download();
    const vec3 glow             = skyViewTexel(sunset, 2, 90);
    EXPECT_GT(glow(0), 1.2f * glow(2)) << glow.transpose();
    // Opposite the sun the sky stays bluer than towards it
    const vec3 away = skyViewTexel(sunset, 10, -90);
    EXPECT_GT(away(2) / std::max(away(0), 1e-6f), glow(2) / std::max(glow(0), 1e-6f));
}

// The moon in daylight gives no direct light in the shading (its light is below 1e-4 of the sun's on the ground: the
// shaders skip its transmittance lookup and GGX); at night, or as bright as the sun, it does. The sky keeps it
TEST_F(AtmosphereTest, MoonSkippedInDaylight)
{
    EnvironmentState day = sunAt(45);
    day.moonDir          = vec3(0, 0.5f, 0.8660254f);
    day.moonIlluminance  = 0.27f;
    Physical(day);
    RenderFrame();
    EXPECT_EQ(scene->lastLightingBlock().moon(3), 0.0f);
    EXPECT_EQ(scene->lastLightingBlock().sun(3), day.sunIlluminance);
    day.moonIlluminance = 0.5f * day.sunIlluminance;
    scene->setEnvironment(day);
    RenderFrame();
    EXPECT_EQ(scene->lastLightingBlock().moon(3), day.moonIlluminance);
    EnvironmentState night = sunAt(-30);
    night.moonDir          = day.moonDir;
    night.moonIlluminance  = 0.27f;
    scene->setEnvironment(night);
    RenderFrame();
    EXPECT_EQ(scene->lastLightingBlock().moon(3), 0.27f);
    EXPECT_EQ(scene->lastLightingBlock().shadowLight(3), 2.0f);
}

// The stars fade out with the sun's elevation between -10 and -4 degrees (the block's intensity: no star loop in a day
// sky; review minor 3)
TEST_F(AtmosphereTest, StarsFadeWithTheSun)
{
    const float intensity = 2;
    for (auto [elevation, expected] : {std::pair<float, float>{10, 0}, {-4, 0}, {-7, 1}, {-10, 2}, {-40, 2}})
    {
        EnvironmentState e = sunAt(elevation);
        e.starIntensity    = intensity;
        Physical(e);
        RenderFrame();
        EXPECT_NEAR(scene->lastLightingBlock().starRotation[0](3), expected, 1e-3f) << elevation;
    }
}

// The night sky: the full moon disc is bright, the dark sky beside it; a moonless sky shows stars (isolated bright
// pixels over a dark background)
TEST_F(AtmosphereTest, NightSkyHasStarsAndMoon)
{
    EnvironmentState night = sunAt(-30, 0);
    night.moonDir          = vec3(0, 0.5f, 0.8660254f);
    night.moonIlluminance  = 0.27f;
    night.ev100            = -1;
    Physical(night, Vec3(night.moonDir.cast<double>()), 0.1);
    const TemplatedImage<ucvec4> moon = Capture(2);
    EXPECT_GT(luminance(moon(height / 2, width / 2)), 200.0f);
    EXPECT_LT(luminance(moon(4, 4)), 60.0f);
    // The disc follows the full moon's illuminance (review minor 4: it was a fixed 0.27 lux): 0 leaves the dark sky
    night.moonFullIlluminance = 0;
    scene->setEnvironment(night);
    const TemplatedImage<ucvec4> none = Capture(1);
    EXPECT_LT(luminance(none(height / 2, width / 2)), 0.5f * luminance(moon(height / 2, width / 2)));
    night.moonFullIlluminance = 0.27f;
    // A moonless night: the moon below the horizon, a wide view up
    night.moonDir         = vec3(0, -0.5f, 0.8660254f);
    night.moonIlluminance = 0;
    night.ev100           = -3;
    Physical(night, Vec3(0, 1, 0.3), 0.3);
    const TemplatedImage<ucvec4> stars = Capture(2);
    std::vector<float> values;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) values.push_back(luminance(stars(y, x)));
    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const float median = sorted[sorted.size() / 2];
    const auto bright  = std::count_if(values.begin(), values.end(), [&](float v) { return v > median + 30; });
    EXPECT_LT(median, 40.0f);
    EXPECT_GT(bright, 10);
    EXPECT_LT(bright, long(values.size() / 10));
}

namespace
{
class EnvironmentMapTest : public AtmosphereTest
{
   protected:
    int CubeResolution() const override { return 128; }
    void TearDown() override
    {
        if (scene) scene->environmentMap().setCaptureOverride(std::nullopt);
        AtmosphereTest::TearDown();
    }
    // The mean colour of a face of the capture cube at a mip
    vec3 captureFace(int face, uint32_t mip)
    {
        TextureCube& cube          = scene->environmentMap().captureCube();
        std::vector<uint8_t> bytes = cube.download(mip, uint32_t(face));
        const int n                = int(cube.mipWidth(mip));
        vec3 sum(0, 0, 0);
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) sum += halfTexel(bytes, x, y, n);
        return sum / float(n * n);
    }
};
}  // namespace

// A new scene's environment map: frames before the first physical frame have the floor as their ambient
// and no environment map; the first capture completes in its own frame (the floor is not shown for several frames)
TEST_F(EnvironmentMapTest, FirstFrameAmbientIsFloor)
{
    EnvironmentState e = sunAt(45);
    e.minAmbient       = vec3(0.5f, 0.25f, 0.125f);
    scene->setEnvironment(e);
    SetCamera(0, Vec3(0, 100, 0), lookRotation(Vec3(0, 0.3, 1)), 1.2);
    Capture(2);
    EXPECT_FALSE(scene->environmentMap().valid());
    EXPECT_EQ(scene->lightingStatistics().sh[0], e.minAmbient);
    Physical(e);
    RenderFrame();
    EXPECT_TRUE(scene->environmentMap().valid());
    EXPECT_EQ(scene->lightingStatistics().environmentMap.captures, 1u);
    EXPECT_EQ(scene->lightingStatistics().environmentMap.progress, 0);
}

// A uniform sky (the capture override) gives a constant SH: band 0 = pi L, the other bands 0
TEST_F(EnvironmentMapTest, UniformSkyGivesConstantSh)
{
    scene->environmentMap().setCaptureOverride(vec3(2, 3, 4));
    Physical(sunAt(45));
    Capture(3);
    scene->environmentMap().harvest();
    const std::array<vec3, 9>& sh = scene->environmentMap().sh();
    for (int c = 0; c < 3; ++c)
    {
        const float expected = 3.14159265f * (2.0f + float(c));
        EXPECT_NEAR(sh[0](c), expected, 0.01f * expected);
    }
    for (int i = 1; i < 9; ++i)
        for (int c = 0; c < 3; ++c) EXPECT_NEAR(sh[size_t(i)](c), 0.0f, 0.01f * sh[0](c)) << "band " << i;
}

// Frames without a change capture nothing; a sun move beyond environmentUpdateDegrees starts one capture
TEST_F(EnvironmentMapTest, RecapturesOnlyWhenDue)
{
    EnvironmentState e = sunAt(30);
    // Settled in one frame (Physical's full update)
    Physical(e);
    RenderFrame();
    const uint64_t captures = scene->lightingStatistics().environmentMap.captures;
    Capture(3);
    EXPECT_EQ(scene->lightingStatistics().environmentMap.captures, captures);
    EXPECT_EQ(scene->lightingStatistics().environmentMap.progress, 0);
    // Below the threshold (0.5 degrees): nothing
    scene->setEnvironment(sunAt(30.3f));
    Capture(2);
    EXPECT_EQ(scene->lightingStatistics().environmentMap.captures, captures);
    scene->setEnvironment(sunAt(31));
    RenderFrame();
    EXPECT_EQ(scene->lightingStatistics().environmentMap.captures, captures + 1);
}

// During a re-capture the bound cube is the previous complete one; it is swapped after all steps (capture, 5 mips, SH)
TEST_F(EnvironmentMapTest, SwapsOnlyCompleteCaptures)
{
    Physical(sunAt(20));
    // Settled in one frame (Physical's full update)
    RenderFrame();
    EnvironmentMap& map       = scene->environmentMap();
    const vk::Image before    = map.frontCube().image();
    const uint64_t completeAt = scene->lightingStatistics().environmentMap.lastCompleteFrame;
    scene->setEnvironment(sunAt(25));
    int frames = 0;
    do
    {
        RenderFrame();
        frames++;
        if (scene->lightingStatistics().environmentMap.progress > 0)
        {
            EXPECT_EQ(map.frontCube().image(), before);
            EXPECT_TRUE(map.valid());
        }
    } while (scene->lightingStatistics().environmentMap.progress > 0 && frames < 20);
    EXPECT_EQ(frames, EnvironmentMap::kMips + 2);
    EXPECT_NE(map.frontCube().image(), before);
    EXPECT_GT(scene->lightingStatistics().environmentMap.lastCompleteFrame, completeAt);
}

// Regression (CodeRabbit #83): a capture in progress when probe_lighting started was swapped in during the sequence,
// so its later views (ambient, specular) described another capture than its frame 1. It completes in frame 1, and the
// front cube stays for the sequence
TEST_F(EnvironmentMapTest, ProbeKeepsOneCapture)
{
    Physical(sunAt(20));
    RenderFrame();
    scene->setEnvironment(sunAt(25));
    RenderFrame();
    ASSERT_TRUE(scene->environmentMap().capturing());
    auto answered = std::make_shared<bool>(false);
    scene->requestProbe(ivec2(64, 64), [answered](const LightingProbe&) { *answered = true; });
    RenderFrame();
    EXPECT_FALSE(scene->environmentMap().capturing());
    const vk::Image front = scene->environmentMap().frontCube().image();
    for (int i = 0; i < 2 * RenderScene::kProbeFrames && !*answered; ++i)
    {
        RenderFrame();
        EXPECT_EQ(scene->environmentMap().frontCube().image(), front) << "frame " << i + 2;
    }
    EXPECT_TRUE(*answered);
}

// Regression (CodeRabbit #84): a requestFullUpdate during a probe's held frames was consumed by a frame that records no
// capture; it now completes a capture in the first frame after the probe
TEST_F(EnvironmentMapTest, FullUpdateDuringProbe)
{
    Physical(sunAt(20));
    RenderFrame();
    const uint64_t captures = scene->lightingStatistics().environmentMap.captures;
    auto answered           = std::make_shared<bool>(false);
    scene->requestProbe(ivec2(64, 64), [answered](const LightingProbe&) { *answered = true; });
    RenderFrame();
    RenderFrame();
    ASSERT_TRUE(scene->probeRunning());
    scene->requestFullUpdate();
    for (int i = 0; i < 2 * RenderScene::kProbeFrames && !*answered; ++i) RenderFrame();
    ASSERT_TRUE(*answered);
    RenderFrame();
    EXPECT_EQ(scene->lightingStatistics().environmentMap.captures, captures + 1);
}

// Below the horizon the capture is the ground (grey, lit through the atmosphere), not the blue sky
TEST_F(EnvironmentMapTest, LowerHemisphereIsGround)
{
    Physical(sunAt(50));
    Capture(2);
    scene->waitIdle();
    const vec3 up = captureFace(2, 2), down = captureFace(3, 2);
    EXPECT_GT(up(2) / up(0), 1.3f * down(2) / down(0)) << up.transpose() << " / " << down.transpose();
    EXPECT_GT(down.sum(), 0.0f);
}

// The sky-view LUT is written only when its inputs change: not in frames with the same environment and camera height,
// again when the sun moves or the camera climbs (review M2: it was rewritten every frame)
TEST_F(EnvironmentMapTest, SkyViewOnlyOnChange)
{
    Physical(sunAt(40));
    RenderFrame();
    const Atmosphere& a = scene->atmosphereLuts();
    const uint64_t n    = a.skyViewComputations();
    RenderFrame();
    RenderFrame();
    EXPECT_EQ(a.skyViewComputations(), n);
    scene->setEnvironment(sunAt(40.5f));
    RenderFrame();
    EXPECT_EQ(a.skyViewComputations(), n + 1);
    // Below the height threshold: nothing; above it: a write with the new horizon
    SetCamera(0, Vec3(0, 105, 0), lookRotation(Vec3(0, 0.3, 1)), 1.2);
    RenderFrame();
    EXPECT_EQ(a.skyViewComputations(), n + 1);
    SetCamera(0, Vec3(0, 2000, 0), lookRotation(Vec3(0, 0.3, 1)), 1.2);
    RenderFrame();
    EXPECT_EQ(a.skyViewComputations(), n + 2);
    EXPECT_FLOAT_EQ(a.skyViewHorizonElevation(), Atmosphere::skyViewHorizon(Atmosphere::radiusOf(2000)));
    EXPECT_FLOAT_EQ(scene->lastLightingBlock().atmosphere(3), a.skyViewHorizonElevation());
}

// The first frame of the physical path after frames without the environment pass (here clay) updates the environment
// map fully: the bound map is of the current environment, not swapped in kMips + 2 frames later (review M4: the old
// capture's ambient showed in the new sky for 7 frames)
TEST_F(EnvironmentMapTest, FullUpdateAfterSkippedFrames)
{
    Physical(sunAt(45));
    RenderFrame();
    ASSERT_TRUE(scene->environmentMap().valid());
    scene->setToneMap(ToneMap::Identity);
    RenderFrame();
    const uint64_t captures = scene->lightingStatistics().environmentMap.captures;
    // The environment changes while the pass is skipped
    scene->setEnvironment(sunAt(8));
    RenderFrame();
    EXPECT_EQ(scene->lightingStatistics().environmentMap.captures, captures);
    scene->setToneMap(ToneMap::Aces);
    RenderFrame();
    const EnvironmentMapStatistics& st = scene->lightingStatistics().environmentMap;
    EXPECT_EQ(st.captures, captures + 1);
    EXPECT_EQ(st.progress, 0);
    EXPECT_EQ(st.lastCompleteFrame, scene->lightingStatistics().frame);
    // Its SH is the one of a fresh full capture of the same environment
    scene->waitIdle();
    scene->environmentMap().harvest();
    const std::array<vec3, 9> first = scene->environmentMap().sh();
    scene->requestFullUpdate();
    RenderFrame();
    scene->waitIdle();
    scene->environmentMap().harvest();
    for (int i = 0; i < 9; ++i)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(first[size_t(i)](c), scene->environmentMap().sh()[size_t(i)](c),
                        1e-3f * std::abs(first[0](c)) + 1e-6f)
                << i << ", " << c;
}

// The camera's horizontal movement re-captures the reflected clouds (500 m), without clouds it does not (review: the
// key held only the cloud drift, so flying without wind kept the clouds of the capture point)
TEST_F(EnvironmentMapTest, RecapturesOnLateralMovement)
{
    EnvironmentState e = sunAt(40);
    e.cloudiness       = 0.5f;
    Physical(e);
    RenderFrame();
    const uint64_t captures = scene->lightingStatistics().environmentMap.captures;
    SetCamera(0, Vec3(300, 100, 0), lookRotation(Vec3(0, 0.3, 1)), 1.2);
    RenderFrame();
    EXPECT_EQ(scene->lightingStatistics().environmentMap.captures, captures);
    SetCamera(0, Vec3(700, 100, 0), lookRotation(Vec3(0, 0.3, 1)), 1.2);
    RenderFrame();
    EXPECT_EQ(scene->lightingStatistics().environmentMap.captures, captures + 1);
    // Without clouds the position does not matter
    e.cloudiness = 0;
    scene->requestFullUpdate();
    scene->setEnvironment(e);
    RenderFrame();
    const uint64_t clear = scene->lightingStatistics().environmentMap.captures;
    SetCamera(0, Vec3(3000, 100, 0), lookRotation(Vec3(0, 0.3, 1)), 1.2);
    RenderFrame();
    EXPECT_EQ(scene->lightingStatistics().environmentMap.captures, clear);
}

// forceFullUpdates: a complete capture (with the swap) in every frame
TEST_F(EnvironmentMapTest, ForceFullUpdatesCapturesEveryFrame)
{
    Physical(sunAt(40), Vec3(0, 0.3, 1), 1.2, true);
    RenderFrame();
    const uint64_t c = scene->lightingStatistics().environmentMap.captures;
    for (int i = 1; i <= 3; ++i)
    {
        RenderFrame();
        EXPECT_EQ(scene->lightingStatistics().environmentMap.captures, c + uint64_t(i));
        EXPECT_EQ(scene->lightingStatistics().environmentMap.progress, 0);
        EXPECT_TRUE(scene->environmentMap().valid());
    }
}

namespace
{
class CloudsTest : public AtmosphereTest
{
   protected:
    void TearDown() override
    {
        if (scene) scene->cloudLayer().setTestBlob(std::nullopt);
        AtmosphereTest::TearDown();
    }
    // A grey plane (a flat box, visible to 1e6 m) with its top at y
    void Plane(MeshId id, float y, float half = 3000)
    {
        MeshLods m            = boxMesh(vec3(half, 1, half), ucvec4(180, 180, 180, 255));
        m.lods[0].endDistance = 1e6f;
        scene->addMaterial(MaterialId(id), "cloud_plane" + std::to_string(id), MaterialData());
        scene->addMesh(id, "cloud_plane" + std::to_string(id), m, MaterialId(id));
        AddSet({instanceAt(vec3(0, y - 1, 0), id)});
    }
    // The physical path with the cloud layer (the cloudiness, altitude), the sun at the elevation from the azimuth
    EnvironmentState Cloudy(float cloudiness, float altitude, float elevationDeg, float azimuthDeg = 90)
    {
        EnvironmentState e = sunAt(elevationDeg, azimuthDeg);
        e.cloudiness       = cloudiness;
        e.clouds.altitude  = altitude;
        e.ev100            = 14;
        return e;
    }
    // The resolved fp16 value of a pixel in the next frame
    vec4 ReadPixel(ivec2 p)
    {
        vec4 value(-1, -1, -1, -1);
        scene->lightingDebugImages().requestPixels({p},
                                                   [&](const std::vector<LightingDebug::PixelValue>& v)
                                                   {
                                                       if (!v.empty() && v[0].valid) value = v[0].color;
                                                   });
        RenderFrame();
        return value;
    }
    void View(LightingDebugView v) { scene->setLightingDebug(LightingDebugState{v, 0}); }
};
}  // namespace

// Regression (CodeRabbit #84): the environment map's key held the weather's cloudiness, not the layer's that the
// capture used: a new seed's capture took the old noise's histogram (it is read back after the frame), and when the
// new histogram changed the layer's cloudiness the capture kept the old threshold until something else changed
TEST_F(CloudsTest, RecapturesWhenTheLayerCloudinessChanges)
{
    EnvironmentState e = Cloudy(0.5f, 2000, 30);
    Physical(e);
    RenderFrame();
    const float before = scene->cloudLayer().effectiveCloudiness(e.cloudiness);
    e.clouds.seed      = 12345;
    scene->setEnvironment(e);
    RenderFrame();
    // The new seed's capture (the old histogram), then its histogram: another layer cloudiness (by 0.014 for this
    // seed) and, when the capture is complete, one more capture
    const uint64_t captures = scene->lightingStatistics().environmentMap.captures;
    for (int i = 0;
         i < 3 * (EnvironmentMap::kMips + 2) && scene->lightingStatistics().environmentMap.captures == captures; ++i)
        RenderFrame();
    ASSERT_GT(std::abs(scene->cloudLayer().effectiveCloudiness(e.cloudiness) - before), 0.01f);
    EXPECT_EQ(scene->lightingStatistics().environmentMap.captures, captures + 1);
}

// The layer's mean coverage follows the weather's cloudiness (the histogram mapping): monotonic, 0 at 0; the GPU
// coverage preview has the same mean
TEST_F(CloudsTest, CoverageFollowsCloudiness)
{
    Physical(Cloudy(0.4f, 2000, 45));
    Capture(2);
    scene->waitIdle();
    CloudLayer& layer = scene->cloudLayer();
    layer.harvest();
    float previous = -1;
    for (int i = 0; i <= 10; ++i)
    {
        const float c    = 0.1f * float(i);
        const float mean = layer.meanCoverage(layer.effectiveCloudiness(c));
        EXPECT_GE(mean, previous) << c;
        EXPECT_NEAR(mean, c, 0.05f) << c;
        previous = mean;
    }
    EXPECT_EQ(layer.meanCoverage(layer.effectiveCloudiness(0)), 0.0f);
    bool got = false;
    LightingDebug::Capture capture;
    PreviewMapping mapping;
    scene->lightingDebugImages().request(LightingImage::CloudCoverage, mapping,
                                         [&](const LightingDebug::Capture& c)
                                         {
                                             capture = c;
                                             got     = true;
                                         });
    RenderFrame();
    ASSERT_TRUE(got);
    EXPECT_EQ(capture.size, ivec2(512, 512));
    EXPECT_NEAR(capture.statistics.mean, scene->lightingStatistics().cloudMeanCoverage, 0.03f);
    EXPECT_NEAR(capture.statistics.mean, 0.4f, 0.05f);
}

// The shadow of a single blob of the layer (a test texture) on a plane moves away from the sun by altitude / tan(el):
// the sun in the east (+x) at 45 degrees and the layer at 500 m put it 500 m west of the blob
TEST_F(CloudsTest, ShadowProjectsAlongSun)
{
    Plane(MeshId(3001), 0);
    scene->cloudLayer().setTestBlob(vec3(0, 0, 10));
    Physical(Cloudy(0.5f, 500, 45, 90), Vec3(0.001, -1, 0), 1.2);
    SetCamera(0, Vec3(0, 1500, 0), lookRotation(Vec3(0.001, -1, 0)), 1.2);
    View(LightingDebugView::CloudShadow);
    const TemplatedImage<ucvec4> img = Capture(3);
    // The centroid of the shadowed pixels
    double sx = 0, sy = 0, n = 0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            if (img(y, x)(0) < 128)
            {
                sx += x;
                sy += y;
                n++;
            }
    ASSERT_GT(n, 4) << "no shadow";
    // The expected ground point (-500, 0, 0) in the image (row 0 at the top)
    const mat4 vp   = scene->viewProjection();
    const vec4 h    = vp * vec4(-500, -1500, 0, 1);
    const double px = (h(0) / h(3) * 0.5 + 0.5) * width, py = (h(1) / h(3) * 0.5 + 0.5) * height;
    EXPECT_NEAR(sx / n, px, 6.0) << "x";
    EXPECT_NEAR(sy / n, py, 6.0) << "y";
    View(LightingDebugView::None);
}

// Seen from above, terrain (here a box) rising above the layer covers it: the box's top is drawn, the layer around it
TEST_F(CloudsTest, OccludedByTerrainAboveAltitude)
{
    MeshLods m            = boxMesh(vec3(60, 250, 60), ucvec4(220, 30, 30, 255));
    m.lods[0].endDistance = 1e6f;
    scene->addMaterial(MaterialId(3002), "cloud_tower", MaterialData());
    scene->addMesh(MeshId(3002), "cloud_tower", m, MaterialId(3002));
    AddSet({instanceAt(vec3(0, 250, 0), MeshId(3002))});
    Plane(MeshId(3003), 0);
    Physical(Cloudy(1, 300, 60));
    SetCamera(0, Vec3(0, 800, 0), lookRotation(Vec3(0.001, -1, 0)), 1.0);
    const TemplatedImage<ucvec4> img = Capture(3);
    const ucvec4 top = img(height / 2, width / 2), cloud = img(8, 8);
    EXPECT_GT(int(top(0)), 2 * int(top(1)) + 20) << "the box above the layer";
    EXPECT_LT(std::abs(int(cloud(0)) - int(cloud(2))), 40) << "the layer around it";
    EXPECT_GT(int(cloud(1)), int(top(1)) + 40);
}

// The layer's alpha fades out over the outer 30% of its radius (no rim)
TEST_F(CloudsTest, EdgeFades)
{
    EXPECT_EQ(CloudLayer::edgeAlpha(0.5f), 1.0f);
    EXPECT_LT(CloudLayer::edgeAlpha(0.95f), 0.2f);
    EXPECT_EQ(CloudLayer::edgeAlpha(1.0f), 0.0f);
    float previous = 2;
    for (int i = 0; i <= 20; ++i)
    {
        const float a = CloudLayer::edgeAlpha(0.05f * float(i));
        EXPECT_LE(a, previous);
        previous = a;
    }
}

// A receiver at or above the layer (2100 m, layer at 2000 m) under full coverage is not shadowed; the ground is
TEST_F(CloudsTest, NoShadowAboveLayer)
{
    Plane(MeshId(3004), 2100, 200);
    Plane(MeshId(3005), 0);
    Physical(Cloudy(1, 2000, 70));
    View(LightingDebugView::CloudShadow);
    SetCamera(0, Vec3(0, 2400, 0), lookRotation(Vec3(0.001, -1, 0)), 0.3);
    Capture(2);
    EXPECT_GT(ReadPixel(ivec2(width / 2, height / 2))(0), 0.99f) << "above the layer";
    SetCamera(0, Vec3(1000, 900, 1000), lookRotation(Vec3(0.001, -1, 0)), 0.3);
    Capture(2);
    EXPECT_LT(ReadPixel(ivec2(width / 2, height / 2))(0), 0.3f) << "below the layer";
    View(LightingDebugView::None);
}

// At 3 degrees the projection reaches about 38 km, beyond the shadow fade: the factor is 1 - opacity * the mean
// coverage everywhere
TEST_F(CloudsTest, LowSunFadesToMean)
{
    Plane(MeshId(3006), 0);
    Physical(Cloudy(0.5f, 2000, 3));
    View(LightingDebugView::CloudShadow);
    SetCamera(0, Vec3(0, 100, 0), lookRotation(Vec3(0.001, -1, 0)), 1.0);
    Capture(2);
    const float mean = scene->lightingStatistics().cloudMeanCoverage;
    EXPECT_NEAR(mean, 0.5f, 0.05f);
    for (ivec2 p : {ivec2(width / 2, height / 2), ivec2(10, 10), ivec2(width - 10, height - 20)})
        EXPECT_NEAR(ReadPixel(p)(0), 1.0f - CloudLayer::kShadowOpacity * mean, 0.02f) << p.transpose();
    View(LightingDebugView::None);
}

// The coverage texture tiles: across its edges the texels differ no more than between neighbours inside (the wrap of
// Weather::cloudOffset to the tile is seamless)
TEST_F(CloudsTest, CoverageTilesSeamlessly)
{
    std::vector<uint8_t> t = scene->cloudLayer().coverageTexture().download(0);
    const int n            = int(CloudLayer::kSize);
    ASSERT_EQ(t.size(), size_t(n) * n);
    double inside = 0, across = 0;
    for (int y = 0; y < n; ++y)
    {
        inside += std::abs(int(t[size_t(y) * n + n / 2]) - int(t[size_t(y) * n + n / 2 - 1]));
        across += std::abs(int(t[size_t(y) * n]) - int(t[size_t(y) * n + n - 1]));
        inside += std::abs(int(t[size_t(n / 2) * n + y]) - int(t[size_t(n / 2 - 1) * n + y]));
        across += std::abs(int(t[y]) - int(t[size_t(n - 1) * n + y]));
    }
    EXPECT_LT(across, 1.5 * inside + n);
}
