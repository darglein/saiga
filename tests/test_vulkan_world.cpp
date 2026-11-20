/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/world/ForestRenderer.h"
#include "saiga/vulkan/world/HeightField.h"
#include "saiga/vulkan/world/WorldBenchmark.h"
#include "saiga/vulkan/world/WorldCommon.h"
#include "saiga/vulkan/world/WorldMcp.h"
#include "saiga/vulkan/world/WorldRenderer.h"

#include <cmath>
#include <cstdlib>

#include "golden_image.h"
#include "mcp_test_helper.h"
#include "vulkan_test_context.h"

using namespace Saiga;
using namespace Saiga::Vulkan;

static HeightField SmallHeightField()
{
    HeightField hf;
    HeightField::Parameters p;
    p.size    = 257;
    p.spacing = 2.0f;
    hf.generate(p);
    return hf;
}

TEST(VulkanWorld, HeightFieldInterpolation)
{
    HeightField hf = SmallHeightField();
    EXPECT_FLOAT_EQ(hf.worldSize(), 512.0f);
    // At the samples the height is exact
    EXPECT_FLOAT_EQ(hf.height(10 * 2.0f, 20 * 2.0f), hf.sample(10, 20));
    // Within a cell: the triangle (i,j) (i+1,j) (i,j+1), as the terrain mesh
    float x = 10.25f * 2, z = 20.5f * 2;
    float ha = hf.sample(10, 20), hb = hf.sample(11, 20), hc = hf.sample(10, 21);
    EXPECT_NEAR(hf.height(x, z), ha + 0.25f * (hb - ha) + 0.5f * (hc - ha), 1e-4f);
    // Normals point up
    EXPECT_GT(hf.normal(100, 100)(1), 0.0f);
    EXPECT_LE(hf.minHeight(), hf.maxHeight());
}

TEST(VulkanWorld, FrustumPlanes)
{
    // Camera at the origin looking along -z
    mat4 proj   = reversedZInfinitePerspective(radians(60.0f), 1.0f, 0.1f);
    auto planes = frustumPlanes(proj);
    EXPECT_TRUE(boxInFrustum(planes, vec3(-1, -1, -11), vec3(1, 1, -9)));
    // Behind the camera
    EXPECT_FALSE(boxInFrustum(planes, vec3(-1, -1, 5), vec3(1, 1, 7)));
    // Far to the side
    EXPECT_FALSE(boxInFrustum(planes, vec3(100, -1, -11), vec3(102, 1, -9)));
    // No far plane: very far away is visible
    EXPECT_TRUE(boxInFrustum(planes, vec3(-1, -1, -1e6f), vec3(1, 1, -1e6f + 2)));

    // Reversed depth: the near plane maps to 1, infinity to 0
    vec4 n = proj * vec4(0, 0, -0.1f, 1);
    EXPECT_NEAR(n(2) / n(3), 1.0f, 1e-5f);
    vec4 f = proj * vec4(0, 0, -1e7f, 1);
    EXPECT_NEAR(f(2) / f(3), 0.0f, 1e-6f);
}

TEST(VulkanWorld, TreePlacement)
{
    HeightField hf = SmallHeightField();
    auto species   = FoliageGenerator::treeSpecies(1);
    auto trees     = ForestRenderer::placeTrees(hf, 500, 3, species);
    EXPECT_GT(trees.size(), 0u);
    EXPECT_LE(trees.size(), 500u);
    for (auto& t : trees)
    {
        float x = t.posYaw(0), z = t.posYaw(2);
        ASSERT_GE(x, 0);
        ASSERT_LE(x, hf.worldSize());
        // On the ground (sunk in a little), above the water
        EXPECT_NEAR(t.posYaw(1), hf.height(x, z) - 0.25f, 1e-3f);
        EXPECT_GT(hf.height(x, z), hf.waterLevel());
        int s = int(t.scaleSpecies(1));
        ASSERT_GE(s, 0);
        ASSERT_LT(s, (int)species.size());
        EXPECT_GE(t.scaleSpecies(0), species[s].heightRange(0));
        EXPECT_LE(t.scaleSpecies(0), species[s].heightRange(1));
    }
    // Deterministic: every tree equal (the normalisation sum is serial, so the scheduling cannot change the placement)
    auto again = ForestRenderer::placeTrees(hf, 500, 3, species);
    ASSERT_EQ(again.size(), trees.size());
    for (size_t i = 0; i < trees.size(); ++i)
    {
        EXPECT_EQ(again[i].posYaw, trees[i].posYaw);
        EXPECT_EQ(again[i].scaleSpecies, trees[i].scaleSpecies);
    }
}

TEST(VulkanWorld, SettingsJson)
{
    WorldSettings s;
    s.treeLod1   = 123;
    s.wireframe  = true;
    s.freezeTime = true;
    WorldSettings r;
    auto unknown = r.fromJson(s.toJson());
    EXPECT_TRUE(unknown.empty());
    EXPECT_EQ(r.treeLod1, 123);
    EXPECT_TRUE(r.wireframe);
    EXPECT_TRUE(r.freezeTime);
    EXPECT_EQ(s.toJson()["shading"].string_value(), "clay");
    EXPECT_TRUE(r.shading == WorldShading::Clay);
    unknown = r.fromJson(json11::Json::object{{"no_such_setting", 1}});
    ASSERT_EQ(unknown.size(), 1u);
    // An unknown shading mode is reported like an unknown key
    unknown = r.fromJson(json11::Json::object{{"shading", "pbr"}});
    ASSERT_EQ(unknown.size(), 1u);
    // Regression: values from MCP were not limited (a tiny grass cell size drew billions of instances)
    r.fromJson(json11::Json::object{{"grass_cell_size", 0.0001}, {"debug_mode", 1e12}, {"fovy", NAN}});
    EXPECT_FLOAT_EQ(r.grassCellSize, 0.1f);
    // Regression: integers above 1000 became 0 instead of the upper limit
    EXPECT_EQ(r.debugMode, 1);
    r.fromJson(json11::Json::object{{"debug_mode", 1001}});
    EXPECT_EQ(r.debugMode, 1);
    r.fromJson(json11::Json::object{{"debug_mode", -5}});
    EXPECT_EQ(r.debugMode, 0);
    EXPECT_FLOAT_EQ(r.fovy, 20.0f);
    // Regression: values of the wrong JSON type were converted silently ("90" -> 0, "true" -> false)
    r.fovy      = 60;
    r.wireframe = true;
    unknown     = r.fromJson(json11::Json::object{{"fovy", "90"}, {"wireframe", "true"}, {"debug_mode", "1"}});
    EXPECT_EQ(unknown.size(), 3u);
    EXPECT_FLOAT_EQ(r.fovy, 60.0f);
    EXPECT_TRUE(r.wireframe);
    // Huge numbers are clamped (no undefined float conversion)
    EXPECT_TRUE(r.fromJson(json11::Json::object{{"tree_lod1", 1e300}}).empty());
    EXPECT_FLOAT_EQ(r.treeLod1, 500.0f);
}

TEST(WorldBenchmark, Summarize)
{
    auto s = WorldBenchmark::summarize({});
    EXPECT_EQ(s.median, 0);
    EXPECT_EQ(s.p95, 0);
    s = WorldBenchmark::summarize({3, 1, 2});
    EXPECT_EQ(s.median, 2);
    EXPECT_EQ(s.p95, 3);
    s = WorldBenchmark::summarize({4, 1, 3, 2});
    EXPECT_EQ(s.median, 2.5);
    // 1..100: the nearest rank of 95% is the 95th value
    std::vector<double> v;
    for (int i = 100; i >= 1; --i) v.push_back(i);
    s = WorldBenchmark::summarize(v);
    EXPECT_EQ(s.median, 50.5);
    EXPECT_EQ(s.p95, 95);
}

TEST(WorldBenchmark, Paths)
{
    // An 8 km world like the default one, at a coarse resolution
    HeightField hf;
    HeightField::Parameters p;
    p.size    = 513;
    p.spacing = 16.0f;
    hf.generate(p);
    float c = hf.worldSize() * 0.5f;
    ASSERT_FLOAT_EQ(c, 4096.0f);
    const int n = WorldBenchmark::measuredFrames;

    for (int i : {0, 150, 300, 599})
    {
        auto pose = WorldBenchmark::pose("flyover", hf, i);
        // On the 3 km circle around the center, 300 m above the ground below, looking at the center on the ground
        EXPECT_NEAR(std::hypot(pose.eye(0) - c, pose.eye(2) - c), 3000.0f, 0.01f);
        EXPECT_NEAR(pose.eye(1), hf.height(pose.eye(0), pose.eye(2)) + 300.0f, 1e-3f);
        EXPECT_NEAR(pose.target(0), c, 1e-3f);
        EXPECT_NEAR(pose.target(2), c, 1e-3f);
        EXPECT_NEAR(pose.target(1), hf.height(c, c), 1e-3f);
    }
    // One revolution in measuredFrames frames
    auto first = WorldBenchmark::pose("flyover", hf, 0), full = WorldBenchmark::pose("flyover", hf, n);
    EXPECT_NEAR((first.eye - full.eye).norm(), 0.0f, 0.05f);
    EXPECT_GT((first.eye - WorldBenchmark::pose("flyover", hf, n / 2).eye).norm(), 5999.0f);

    auto start = WorldBenchmark::pose("forest", hf, 0), end = WorldBenchmark::pose("forest", hf, n - 1);
    EXPECT_NEAR(start.eye(0), c - 300.0f, 1e-3f);
    EXPECT_NEAR(end.eye(0), c + 300.0f, 1e-3f);
    for (int i : {0, 200, n - 1})
    {
        auto pose = WorldBenchmark::pose("forest", hf, i);
        EXPECT_NEAR(pose.eye(2), c, 1e-3f);
        float ground = std::max(hf.height(pose.eye(0), c), hf.waterLevel());
        EXPECT_NEAR(pose.eye(1), ground + 2.0f, 1e-3f);
        // Looking along +x
        vec3 dir = (pose.target - pose.eye).normalized();
        EXPECT_NEAR(dir(0), 1.0f, 1e-6f);
    }
    EXPECT_TRUE(WorldBenchmark::known("forest"));
    EXPECT_FALSE(WorldBenchmark::known("forests"));
}

// Regression: a minimized window or a swap chain recreation during the measured frames inflated p95 unnoticed
TEST(WorldBenchmark, CountStalls)
{
    std::vector<double> v(600, 10.0);
    EXPECT_EQ(WorldBenchmark::countStalls(v), 0);
    v[100] = 2000;  // minimized for 2 s
    v[200] = 45;    // 4.5 x the median, but not longer than 50 ms
    EXPECT_EQ(WorldBenchmark::countStalls(v), 1);
    // A slow device: 4 x the median counts, not the 50 ms
    std::vector<double> slow(600, 200.0);
    slow[5] = 700;
    EXPECT_EQ(WorldBenchmark::countStalls(slow), 0);
    slow[6] = 900;
    EXPECT_EQ(WorldBenchmark::countStalls(slow), 1);
    EXPECT_EQ(WorldBenchmark::countStalls({}), 0);
}

TEST(WorldBenchmark, ResultJson)
{
    // No init: the benchmark only registers a frame callback, the samples are added directly
    WorldRenderer world;
    WorldBenchmark bench("forest", world, "test gpu");
    EXPECT_EQ(world.frameCallbacks.size(), 1u);
    const int warmup = WorldBenchmark::warmupFrames, n = WorldBenchmark::measuredFrames;

    // CPU frame f has the time f ms, the GPU times of frame f arrive 2 frames later: pass p has (p + 1) * f / 100 ms
    for (int frame = 1; frame <= warmup + n + 2; ++frame)
    {
        WorldRenderer::FrameTimes t;
        t.cpuMs     = frame;
        t.cpuWorkMs = frame * 0.5;
        t.cpuFrame  = frame;
        if (frame > 2)
        {
            t.gpuFrame = frame - 2;
            for (int p = 0; p < WorldRenderer::PASS_COUNT; ++p) t.gpuPassMs[p] = (p + 1) * t.gpuFrame / 100.0;
            t.gpuMs = 0;
            for (double ms : t.gpuPassMs) t.gpuMs += ms;
        }
        EXPECT_FALSE(bench.finished()) << frame;
        bench.addSample(t);
    }
    EXPECT_TRUE(bench.finished());

    auto r = bench.result();
    EXPECT_EQ(r["gpu"].string_value(), "test gpu");
    EXPECT_EQ(r["quality"].string_value(), "default");
    // Only the measured frames 61..660: the warm-up frames are excluded
    EXPECT_EQ(r["frames"].int_value(), n);
    EXPECT_EQ(r["gpu_frames"].int_value(), n);
    EXPECT_DOUBLE_EQ(r["cpu_ms"]["median"].number_value(), warmup + 300.5);
    EXPECT_DOUBLE_EQ(r["cpu_ms"]["p95"].number_value(), warmup + 570);
    EXPECT_DOUBLE_EQ(r["cpu_work_ms"]["median"].number_value(), (warmup + 300.5) * 0.5);
    EXPECT_TRUE(r["build"]["type"].is_string());
    EXPECT_TRUE(r["build"]["asserts"].is_bool());
    EXPECT_TRUE(r["build"]["archnative"].is_bool());
    EXPECT_EQ(r["stalled_frames"].int_value(), 0);
    for (int p = 0; p < WorldRenderer::PASS_COUNT; ++p)
    {
        auto pass = r["gpu_ms"][WorldRenderer::gpuPassName(p)];
        ASSERT_TRUE(pass.is_object()) << WorldRenderer::gpuPassName(p);
        EXPECT_NEAR(pass["median"].number_value(), (p + 1) * (warmup + 300.5) / 100.0, 1e-9);
    }

    // Without GPU samples (no timestamps) gpu_ms is null instead of zeros that look like a fast result
    WorldBenchmark noGpu("forest", world, "test gpu");
    for (int frame = 1; frame <= warmup + n; ++frame)
    {
        WorldRenderer::FrameTimes t;
        t.cpuMs    = frame;
        t.cpuFrame = frame;
        noGpu.addSample(t);
    }
    EXPECT_TRUE(noGpu.result()["gpu_ms"].is_null());
    EXPECT_EQ(noGpu.result()["gpu_frames"].int_value(), 0);
}

// A forward renderer with MSAA and reversed depth, the configuration of the world renderer. A separate fixture
// instead of VulkanTest because the window needs ForwardRenderingParameters, which VulkanTest does not take.
class VulkanWorldRender : public ::testing::Test
{
   protected:
    // The size of the rendered images
    virtual ivec2 imageSize() const { return ivec2(96, 64); }

    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
        WindowParameters windowParameters;
        windowParameters.width  = imageSize()(0);
        windowParameters.height = imageSize()(1);
        windowParameters.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
        Vulkan::VulkanParameters vulkanParameters;
        vulkanParameters.enableImgui = false;
        ForwardRenderingParameters params;
        params.msaaSamples = 4;
        params.reversedZ   = true;
        window             = std::make_unique<Vulkan::HeadlessWindow>(windowParameters);
        renderer           = std::make_unique<Vulkan::VulkanForwardRenderer>(*window, vulkanParameters, params);
        // The forest's indirect draws need it (asserted in ForestRenderer::init); Vulkan 1.3 does not require it
        if (!renderer->base().enabledFeatures.drawIndirectFirstInstance)
            GTEST_SKIP() << "drawIndirectFirstInstance is not supported";
        renderer->setRenderObject(app);
        window->setUpdateObject(app);
    }
    void TearDown() override
    {
        if (renderer) renderer->waitIdle();
        app = VulkanTestApp();
        renderer.reset();
        window.reset();
    }

    TemplatedImage<ucvec4> RenderFrame()
    {
        TemplatedImage<ucvec4> result;
        renderer->requestScreenshot([&](const TemplatedImage<ucvec4>& img) { result = img; });
        window->renderFrames(1);
        return result;
    }

    // The world in the forward renderer's pass (reversed depth)
    void Init(WorldRenderer& world, WorldParameters params, std::shared_ptr<const HeightField> heightField = nullptr)
    {
        params.reversedZ = renderer->parameters().reversedZ;
        if (heightField)
            world.init(renderer->base(), renderer->renderingFormats(), params, std::move(heightField));
        else
            world.init(renderer->base(), renderer->renderingFormats(), params);
    }

    VulkanTestApp app;
    std::unique_ptr<Vulkan::HeadlessWindow> window;
    std::unique_ptr<Vulkan::VulkanForwardRenderer> renderer;
};

TEST_F(VulkanWorldRender, RendersTerrainTreesAndSky)
{
    EXPECT_TRUE(renderer->parameters().reversedZ);
    EXPECT_EQ(renderer->clearDepth(), 0.0f);

    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 513;
    params.treeCount    = 2000;
    Init(world, params);

    // Above the center of the world, looking down at the horizon
    auto& hf     = world.heightField();
    vec3 c       = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    c(1)         = hf.height(c(0), c(2));
    vec3 eye     = c + vec3(0, 60, 200);
    mat4 view    = lookAt(eye, c + vec3(0, 20, 0), vec3(0, 1, 0));
    float aspect = 96.0f / 64.0f;

    app.onTransfer = [&](vk::CommandBuffer cmd) { world.transfer(cmd, view, eye, aspect); };
    app.onRender   = [&](vk::CommandBuffer cmd) { world.render(cmd); };

    // Without the sky pass the top of the image is the clear color
    world.settings.sky           = false;
    TemplatedImage<ucvec4> noSky = RenderFrame();
    world.settings.sky           = true;

    TemplatedImage<ucvec4> img;
    // A few frames: the statistics are read back from completed frames
    for (int i = 0; i < 5; ++i) img = RenderFrame();
    ASSERT_EQ(img.w, 96);

    // The sky at the top is a uniform gray that differs from the clear color (regression: the test passed without a
    // sky pass, because the clear color is gray as well), the terrain at the bottom differs from the clear color
    ucvec4 top = img(1, 48), clearTop = noSky(1, 48), bottom = img(62, 48);
    EXPECT_EQ(top(0), top(1));
    EXPECT_EQ(top(1), top(2));
    EXPECT_GT(std::abs(int(top(0)) - int(clearTop(0))), 2) << "sky " << int(top(0)) << " clear " << int(clearTop(0));
    EXPECT_GT(std::abs(int(bottom(0)) - int(clearTop(0))), 2);
    EXPECT_GT(world.stats().terrainNodes, 0);
    EXPECT_GT(world.stats().treesVisible[0] + world.stats().treesVisible[1] + world.stats().treesVisible[2], 0);

    renderer->waitIdle();
    app = VulkanTestApp();
}

// An external height field (S13b: generated once for a game's simulation and its renderer) is shared, not copied, and
// renders like a generated one
TEST_F(VulkanWorldRender, SharesAnExternalHeightField)
{
    HeightField::Parameters terrain;
    terrain.size  = 257;
    auto external = std::make_shared<HeightField>();
    external->generate(terrain);

    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 129;  // ignored with an external height field
    params.treeCount    = 500;
    Init(world, params, external);
    EXPECT_EQ(&world.heightField(), external.get());
    EXPECT_EQ(world.heightField().size(), 257);
    EXPECT_EQ(external.use_count(), 2);  // here and in the world renderer
    auto& hf       = world.heightField();
    vec3 c         = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    c(1)           = hf.height(c(0), c(2));
    vec3 eye       = c + vec3(0, 60, 200);
    mat4 view      = lookAt(eye, c + vec3(0, 20, 0), vec3(0, 1, 0));
    float aspect   = 96.0f / 64.0f;
    app.onTransfer = [&](vk::CommandBuffer cmd) { world.transfer(cmd, view, eye, aspect); };
    app.onRender   = [&](vk::CommandBuffer cmd) { world.render(cmd); };
    for (int i = 0; i < 3; ++i) RenderFrame();
    EXPECT_GT(world.stats().terrainNodes, 0);
    renderer->waitIdle();
    app = VulkanTestApp();
    world.destroy();
    EXPECT_EQ(external.use_count(), 1);  // released by destroy()
}

// Regression: positions were passed to the fragment shader in world space as floats. Far from the origin (x = 7 km:
// float steps of 0.5 mm) and close to the ground, dFdx/dFdy of the position (flat shading) were rounding noise and
// the ground showed salt-and-pepper noise. The world passes now work with positions relative to the camera.
TEST_F(VulkanWorldRender, FlatShadingIsStableFarFromTheOrigin)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size    = 2049;
    params.terrain.spacing = 4;  // 8 km world
    params.treeCount       = 10;
    Init(world, params);
    world.settings.grass       = false;
    world.settings.trees       = false;
    world.settings.fog         = false;
    world.settings.flatShading = true;
    world.settings.fovy        = 20;

    auto& hf = world.heightField();
    // 15 cm above a point far from the origin that is above the water (x ~ 7 km: float steps of 0.5 mm; with the narrow
    // field of view a pixel covers less than 1 mm of ground, like a full HD window 0.3 m above the ground)
    vec3 eye(0, 0, 0);
    for (float x = hf.worldSize() * 0.85f; x > hf.worldSize() * 0.5f; x -= 16)
    {
        eye = vec3(x, hf.height(x, x) + 0.15f, x);
        if (hf.height(x, x) > hf.waterLevel() + 2) break;
    }
    // Looking down at the ground in front of the camera
    mat4 view      = lookAt(eye, eye + vec3(0.6f, -1.0f, 0.4f), vec3(0, 1, 0));
    float aspect   = 96.0f / 64.0f;
    app.onTransfer = [&](vk::CommandBuffer cmd) { world.transfer(cmd, view, eye, aspect); };
    app.onRender   = [&](vk::CommandBuffer cmd) { world.render(cmd); };

    // Reference without terrain and water: the check below must see terrain, not a uniform background
    world.settings.terrain           = false;
    world.settings.water             = false;
    TemplatedImage<ucvec4> reference = RenderFrame();
    // Water stays off: its uniform plane could pass both checks without any terrain
    world.settings.terrain = true;

    TemplatedImage<ucvec4> img = RenderFrame();
    ASSERT_EQ(img.w, 96);
    ASSERT_EQ(reference.w, 96);
    int differs = 0, pixels = 0;
    for (int y = 8; y < 56; ++y)
    {
        for (int x = 8; x < 88; ++x)
        {
            differs += std::abs(int(img(y, x)(0)) - int(reference(y, x)(0))) > 2;
            pixels++;
        }
    }
    EXPECT_GT(differs, pixels * 9 / 10) << "the terrain does not cover the image";

    // The ground in front of the camera is one or two terrain triangles: neighbouring pixels have about the same
    // shade (with the bug about half of them differed by more than 20)
    int rough = 0, total = 0;
    for (int y = 8; y < 56; ++y)
    {
        for (int x = 8; x < 87; ++x)
        {
            rough += std::abs(int(img(y, x)(0)) - int(img(y, x + 1)(0))) > 20;
            total++;
        }
    }
    EXPECT_LT(rough, total / 50) << rough << " of " << total << " neighbouring pixels differ by more than 20";

    renderer->waitIdle();
    app = VulkanTestApp();
}

// Golden images of the prototype world (docs/engine_design.md §17.6): 512 x 512, clay shading, frozen time, a 1 km
// world with 5000 trees, seed 1, captured after 5 frames. They gate on lavapipe only; other devices print the metrics.
// The drawn instance and terrain node counts gate on lavapipe only as well: the tree placement uses std random
// distributions and float comparisons, so other standard libraries, compilers (FMA, -march=native) or drivers may
// place or cull a few trees differently.
// S17: they are rendered through RenderScene's frame (clay mode: the hardware average resolve, the identity tone map),
// one render per golden.
class VulkanWorldGolden : public ::testing::Test
{
   protected:
    static constexpr int size = 512;

    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
        WindowParameters windowParameters;
        windowParameters.width  = size;
        windowParameters.height = size;
        windowParameters.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
        Vulkan::VulkanParameters vulkanParameters;
        vulkanParameters.enableImgui = false;
        RenderSceneParameters params;
        params.imgui       = false;
        params.msaaSamples = 4;
        window             = std::make_unique<Vulkan::HeadlessWindow>(windowParameters);
        scene              = std::make_unique<RenderScene>(*window, vulkanParameters, params);
        // The forest's indirect draws need it (asserted in ForestRenderer::init); Vulkan 1.3 does not require it
        if (!scene->base().enabledFeatures.drawIndirectFirstInstance)
            GTEST_SKIP() << "drawIndirectFirstInstance is not supported";
    }
    void TearDown() override
    {
        if (scene) scene->waitIdle();
        scene.reset();
        window.reset();
    }

    // .data(): MSVC cannot convert vk::ArrayWrapper1D<char, 256> to std::string directly
    std::string deviceName() const
    {
        return std::string(scene->base().physicalDevice.getProperties().deviceName.data());
    }

    // Renders 5 frames from the camera and returns the last one
    TemplatedImage<ucvec4> Capture(WorldRenderer& world, const vec3& eye, const vec3& target)
    {
        mat4 view    = lookAt(eye, target, vec3(0, 1, 0));
        float aspect = 1.0f;
        TemplatedImage<ucvec4> img;
        auto transfer = [&](vk::CommandBuffer cmd)
        {
            world.setOutputLinear(scene->sceneWritesLinear(), scene->preExposure());
            world.transfer(cmd, view, eye, aspect);
        };
        auto render = [&](vk::CommandBuffer cmd) { world.render(cmd); };
        for (int i = 0; i < 5; ++i)
        {
            if (i == 4) scene->requestScreenshot([&](const TemplatedImage<ucvec4>& shot) { img = shot; });
            scene->renderFrame(transfer, render, nullptr);
        }
        return img;
    }

    void InitWorld(WorldRenderer& world)
    {
        WorldParameters params;
        params.terrain.size = 513;  // 2 m spacing: 1 km
        params.treeCount    = 5000;
        params.seed         = 1;
        params.reversedZ    = true;
        world.init(scene->base(), scene->formats(), params);
        world.settings.shading    = WorldShading::Clay;
        world.settings.freezeTime = true;
    }

    // The committed counts (docs/engine_design.md §17.6; S19 compares its culling against them)
    void ExpectCounts(WorldRenderer& world, const char* name, int lod0, int lod1, int lod2, int terrainNodes)
    {
        auto& st = world.stats();
        std::cout << name << " trees per lod: " << st.treesVisible[0] << " " << st.treesVisible[1] << " "
                  << st.treesVisible[2] << ", terrain nodes " << st.terrainNodes << std::endl;
        // Not checked while the goldens are regenerated: the printed counts replace the committed ones
        if (!goldenGating(deviceName()) || goldenUpdateMode()) return;
        EXPECT_EQ(st.treesVisible[0], lod0);
        EXPECT_EQ(st.treesVisible[1], lod1);
        EXPECT_EQ(st.treesVisible[2], lod2);
        EXPECT_EQ(st.terrainNodes, terrainNodes);
    }

    // The frames in flight use the world: wait for them before the world is destroyed
    void Finish() { scene->waitIdle(); }

    std::unique_ptr<Vulkan::HeadlessWindow> window;
    std::unique_ptr<RenderScene> scene;
};

TEST_F(VulkanWorldGolden, WorldOverview)
{
    WorldRenderer world;
    InitWorld(world);
    auto& hf = world.heightField();
    // 300 m above the ground 100 m from the west edge (at the center z), looking 45 degrees down along +x: the terrain
    // and the trees fill the image, only a thin band of sky and the far edge of the world remain at the top
    vec3 c(100, 0, hf.worldSize() * 0.5f);
    c(1)     = hf.height(c(0), c(2));
    vec3 eye = c + vec3(0, 300, 0);
    vec3 dir(std::cos(radians(45.0f)), -std::sin(radians(45.0f)), 0);
    auto img = Capture(world, eye, eye + dir);
    // Not ASSERT: an early return would destroy the world while frames are in flight (compareGolden also reports a
    // size mismatch)
    EXPECT_EQ(img.w, 512);
    compareGolden(img, "world_overview", deviceName());
    ExpectCounts(world, "world_overview", 0, 7, 3575, 45);
    Finish();
}

TEST_F(VulkanWorldGolden, WorldForest)
{
    WorldRenderer world;
    InitWorld(world);
    auto& hf = world.heightField();
    // 2 m above the ground at the edge of a clearing 28 m west and 80 m north of the center (the nearest tree is 15 m
    // away), looking along +x and slightly down: a close tree with alpha-to-coverage foliage, grass, trunks, the
    // forest at mid distance and the sky
    vec3 eye(hf.worldSize() * 0.5f - 28, 0, hf.worldSize() * 0.5f - 80);
    eye(1)   = std::max(hf.height(eye(0), eye(2)), hf.waterLevel()) + 2.0f;
    auto img = Capture(world, eye, eye + vec3(1, -0.2f, 0));
    // Not ASSERT: an early return would destroy the world while frames are in flight (compareGolden also reports a
    // size mismatch)
    EXPECT_EQ(img.w, 512);
    compareGolden(img, "world_forest", deviceName());

    ExpectCounts(world, "world_forest", 43, 281, 677, 38);
    Finish();
}

TEST_F(VulkanWorldRender, FrameTimes)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 257;
    params.treeCount    = 500;
    Init(world, params);
    auto& hf       = world.heightField();
    vec3 c         = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    vec3 eye       = c + vec3(0, hf.height(c(0), c(2)) + 80, 100);
    mat4 view      = lookAt(eye, c, vec3(0, 1, 0));
    app.onTransfer = [&](vk::CommandBuffer cmd) { world.transfer(cmd, view, eye, 1.5f); };
    app.onRender   = [&](vk::CommandBuffer cmd) { world.render(cmd); };

    // The times of a frame are delivered in a later transfer(): CPU times one frame, GPU times a few frames later
    int gpuSamples = 0;
    std::vector<int> gpuFrames;
    for (int i = 0; i < 10; ++i)
    {
        RenderFrame();
        auto& t = world.lastFrameTimes();
        EXPECT_EQ(t.cpuFrame, world.frameNumber() - 1);
        if (world.frameNumber() > 1)
        {
            EXPECT_GT(t.cpuMs, 0);
            EXPECT_GT(t.cpuWorkMs, 0);
        }
        if (t.gpuMs >= 0)
        {
            gpuSamples++;
            gpuFrames.push_back(t.gpuFrame);
            EXPECT_LT(t.gpuFrame, world.frameNumber());
            EXPECT_GT(t.gpuFrame, 0);
            double sum = 0;
            for (double ms : t.gpuPassMs) sum += ms;
            EXPECT_NEAR(sum, t.gpuMs, 1e-6);
        }
    }
    // Some devices have no timestamps on the queue (statistics show -1 then)
    if (world.stats().gpuMs >= 0)
    {
        EXPECT_GT(gpuSamples, 0);
        // Regression: the readback only took the newest completed frame and lost about half of the frames. Every
        // completed frame is delivered, in order and without gaps.
        for (size_t i = 1; i < gpuFrames.size(); ++i) EXPECT_EQ(gpuFrames[i], gpuFrames[i - 1] + 1) << i;
        EXPECT_GE(gpuSamples, 6);
    }

    renderer->waitIdle();
    app = VulkanTestApp();
}

// ---------------------------------------------------------------------------------------------------------------------
// S14: the game tools of the world prototype (game_world) through handleMessage

namespace
{
McpServer::Json callWorldTool(McpServer& server, const std::string& name, const McpServer::Json& args)
{
    Poller poller(server);
    return server.handleMessage(McpRequest("tools/call", McpServer::Json::object{{"name", name}, {"arguments", args}}));
}
McpServer::Json toolResultJson(const McpServer::Json& r)
{
    std::string err;
    return McpServer::Json::parse(r["result"]["content"][0]["text"].string_value(), err);
}
}  // namespace

TEST_F(VulkanWorldRender, GameTools)
{
    using Json = McpServer::Json;
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 129;
    params.treeCount    = 100;
    Init(world, params);
    PerspectiveCamera camera;
    camera.setView(lookAt(vec3(10, 50, 10), vec3(20, 40, 20), vec3(0, 1, 0)));
    bool walkMode   = false;
    float eyeHeight = 1.7f;
    McpServer server("test");
    Vulkan::WorldMcpContext context;
    context.world     = &world;
    context.camera    = &camera;
    context.walkMode  = &walkMode;
    context.eyeHeight = &eyeHeight;
    Vulkan::registerWorldMcpTools(server, context);
    const HeightField& hf = world.heightField();

    // game_get_state: the extent, statistics, settings and the camera's ground state
    Json s = toolResultJson(callWorldTool(server, "game_get_state", Json::object{}));
    EXPECT_EQ(s["world"]["size_m"].number_value(), hf.worldSize()) << s.dump();
    EXPECT_TRUE(s["settings"].is_object());
    EXPECT_TRUE(s["camera"]["ground_height"].is_number());
    EXPECT_FALSE(s["camera"]["walk_mode"].bool_value());

    // game_set_settings: valid, clamped and invalid keys
    s = toolResultJson(callWorldTool(server, "game_set_settings", Json::object{{"fovy", 1000.0}, {"nope", 1}}));
    EXPECT_LT(s["settings"]["fovy"].number_value(), 1000) << s.dump();  // clamped to the GUI range
    EXPECT_EQ(s["invalid_keys"][0].string_value(), "nope") << s.dump();

    // game_terrain_query: the field's heights, limits and errors
    const float x = hf.worldSize() * 0.25f, z = hf.worldSize() * 0.75f;
    // Nested arrays built explicitly: Json::array{Json::array{...}} is a copy, not a nesting, with Clang
    Json::array points;
    points.push_back(Json::array{x, z});
    Json q = toolResultJson(callWorldTool(server, "game_terrain_query", Json::object{{"points", points}}));
    EXPECT_NEAR(q[0]["height"].number_value(), hf.height(x, z), 0.01) << q.dump();
    Json::array badPoints;
    badPoints.push_back(Json::array{1});
    EXPECT_TRUE(callWorldTool(server, "game_terrain_query", Json::object{{"points", badPoints}})["result"]["isError"]
                    .bool_value());

    // game_set_camera: [x, z] with a height above ground keeps the view direction; walk mode on and off
    const vec3 dir = -make_vec3(camera.model.col(2)).normalized();
    Json c         = toolResultJson(callWorldTool(
        server, "game_set_camera", Json::object{{"position", Json::array{x, z}}, {"height_above_ground", 30.0}}));
    EXPECT_NEAR(c["height_above_ground"].number_value(), 30, 0.02) << c.dump();
    EXPECT_NEAR(c["ground_height"].number_value(), hf.height(x, z), 0.02) << c.dump();
    EXPECT_NEAR(camera.getPosition()(0), x, 1e-2);
    EXPECT_LT((-make_vec3(camera.model.col(2)).normalized() - dir).norm(), 1e-3);
    c = toolResultJson(callWorldTool(server, "game_set_camera", Json::object{{"walk_mode", true}}));
    EXPECT_TRUE(walkMode);
    EXPECT_NEAR(c["height_above_ground"].number_value(), eyeHeight, 0.02) << c.dump();
    callWorldTool(server, "game_set_camera", Json::object{{"walk_mode", false}});
    EXPECT_FALSE(walkMode);
    EXPECT_TRUE(
        callWorldTool(server, "game_set_camera", Json::object{{"position", Json::array{1}}})["result"]["isError"]
            .bool_value());
    EXPECT_TRUE(
        callWorldTool(server, "game_set_camera", Json::object{{"yaw_deg", 3}})["result"]["isError"].bool_value());
    world.destroy();
}
