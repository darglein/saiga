/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/ForestRenderer.h"
#include "saiga/vulkan/world/HeightField.h"
#include "saiga/vulkan/world/WorldCommon.h"
#include "saiga/vulkan/world/WorldRenderer.h"

#include <cmath>
#include <cstdlib>

#include "golden_image.h"
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
    world.init(*renderer, params);

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
    world.init(*renderer, params);
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
class VulkanWorldGolden : public VulkanWorldRender
{
   protected:
    ivec2 imageSize() const override { return ivec2(512, 512); }

    std::string deviceName() const { return std::string(renderer->base().physicalDevice.getProperties().deviceName); }

    // Renders 5 frames from the camera and returns the last one
    TemplatedImage<ucvec4> Capture(WorldRenderer& world, const vec3& eye, const vec3& target)
    {
        mat4 view      = lookAt(eye, target, vec3(0, 1, 0));
        float aspect   = float(imageSize()(0)) / float(imageSize()(1));
        app.onTransfer = [&world, view, eye, aspect](vk::CommandBuffer cmd) { world.transfer(cmd, view, eye, aspect); };
        app.onRender   = [&world](vk::CommandBuffer cmd) { world.render(cmd); };
        TemplatedImage<ucvec4> img;
        for (int i = 0; i < 5; ++i) img = RenderFrame();
        return img;
    }

    void InitWorld(WorldRenderer& world)
    {
        WorldParameters params;
        params.terrain.size = 513;  // 2 m spacing: 1 km
        params.treeCount    = 5000;
        params.seed         = 1;
        world.init(*renderer, params);
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

    // The callbacks reference the world: wait for the frames in flight and drop them before the world is destroyed
    void Finish()
    {
        renderer->waitIdle();
        app = VulkanTestApp();
    }
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
