/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lighting of S36 on RenderScene (docs/stages/S36_lighting.plan.md, Tests): a1b the LIGHTING_PBR variant (the BRDF
// LUT, energy conservation, the studio environment of clay mode, the prototype colours) and the debug core. One scene
// per test suite (RenderSceneSuiteTest), skipped without a Vulkan 1.3 device.

#include "saiga/vulkan/scene/BrdfLut.h"
#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/SkyAmbient.h"

#include <algorithm>
#include <cmath>
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
    // Computed once per process (the later scenes upload it; it cost about 0.1 s per RenderScene on lavapipe)
    EXPECT_EQ(BrdfLut::computations(), 1);
}

// The probe's radiance against pbr.glsl in closed form at the centre of a sphere (NdotV ~ 1):
// - a grey metal of roughness 1 under a uniform environment of radiance 1: the specular part is F0 A + B and the
//   diffuse part the multi-scattering Fms Ems of the LUT's corner texel (NdotV and roughness at the LUT's upper edge).
//   Regression: the LUT was sampled with the default repeat address mode, so at NdotV ~ 1 and roughness 1 the filter
//   mixed in the texels of NdotV ~ 0 and roughness ~ 0 (about half the weight);
// - a grey dielectric of roughness 0.5 in sunlight (no ambient): the direct radiance of GGX, Smith, Schlick and Lambert
//   with the energy compensation;
// - a white metal of roughness 0.2 at its highlight under 100 000 lux (about 5e6 cd/m^2): finite. Regression: the raw
//   probe frames wrote cd/m^2 without the pre-exposure into the fp16 target (inf above 65504)
TEST_F(Pbr, ProbeMatchesClosedForm)
{
    UsePbr();
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
            EXPECT_NEAR(p.direct(c), expect(c), (k == 0 ? 0.03f : 0.05f) * expect(c))
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

// ---------------------------------------------------------------------------------------------------------------------
// The debug core (S36a1b): views, overlays, previews, readbacks, the probe

namespace
{
class LightingDebugTest : public RenderSceneSuiteTest
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

// Every view and the overlays render 2 frames without a validation error (the LIGHTING_PBR variant, clay and ACES with
// an environment); the Normals of a plane facing +y read (0.5, 1, 0.5) from the resolved fp16 image (raw data)
TEST_F(LightingDebugTest, EveryViewRenders)
{
    UsePbr();
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
    UsePbr();
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
    // The variant switch itself recreates them (the reference that the counter counts)
    UsePbr(false);
    Capture(1);
    EXPECT_GT(scene->scenePipelineCreations(), before);
}

// A debug frame (Identity latched) does not update the pre-exposure history: the frame after it has the pre-exposure
// of the frame before it (with an effective exposure within kMaxPreExposureChange of 1, so a reset to 1 would show)
TEST_F(LightingDebugTest, DebugFrameKeepsPreExposure)
{
    UsePbr();
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
    UsePbr();
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
    UsePbr();
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
    UsePbr();
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
    UsePbr();
    PlaneScene(420);
    Capture(2);
    LightingDebug& d = scene->lightingDebugImages();
    PreviewMapping linear, log2Mapping, tab;
    log2Mapping.log = true;
    tab.log         = true;
    tab.colorMap    = ColorMap::Turbo;
    std::optional<LightingDebug::Capture> a, b;
    d.request(LightingImage::Luminance, linear, [&](const LightingDebug::Capture& c) { a = c; });
    d.request(LightingImage::Luminance, log2Mapping, [&](const LightingDebug::Capture& c) { b = c; });
    d.show(LightingImage::Luminance, tab);
    for (int i = 0; i < 4 && !(a && b); ++i) RenderFrame();
    ASSERT_TRUE(a && b);
    EXPECT_GT(a->statistics.max, 0.05f);
    EXPECT_NEAR(b->statistics.max, std::log2(a->statistics.max), 0.01f);
    // Gray, not the tab's turbo
    const ucvec4 c = a->image(height / 2, width / 2 - 20);
    EXPECT_EQ(c(0), c(1));
    EXPECT_EQ(c(1), c(2));
}

// While a probe runs, a screenshot and a capture requested with it and a capture requested during it wait for the frame
// after the sequence, so they read a normal frame. Regression: they read the probe's debug frames (an Albedo / Normals
// / raw radiance image); a capture requested with the probe was converted in its frame 1 (CodeRabbit #82)
TEST_F(LightingDebugTest, ProbeDefersScreenshotsAndCaptures)
{
    UsePbr();
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
    UsePbr();
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
}

// probe_lighting's sequence: 7 frames at a pixel of the lit plane, the values equal the views' at that pixel, the
// camera and the environment are held (a setEnvironment during it takes effect after it), consistent
TEST_F(LightingDebugTest, ProbeMatchesViews)
{
    UsePbr();
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
