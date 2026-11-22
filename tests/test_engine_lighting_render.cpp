/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lighting goldens of S36 (docs/stages/S36_lighting.plan.md, Tests: Goldens; docs/engine_design.md §17.6): a World
// with the environment enabled (WorldTime, Weather) on a VulkanRenderSink, the mountainRidgeHook terrain of a 4 km
// space at base spacing 2, 200 PrototypeFoliage trees and a few flat-colour boxes, 512 x 512 with 4 samples, ACES with
// a fixed EV100 per golden, the environment map fully updated in the captured frame (golden mode), the environment
// from the world (environmentState; its wind phases are 0 at time 0); one scene and world for the suite. S36a1:
// without sun shadows (a2 regenerates them with the cascades and the horizon map). They gate on lavapipe only; other
// devices print the metrics. Regenerate on lavapipe: SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_lighting_render
// --gtest_filter='LightingGolden.*'

#include "saiga/engine/core/Log.h"
#include "saiga/engine/lighting/EnvironmentExtract.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/sync/VulkanRenderSink.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/World.h"
#include "saiga/vulkan/world/FoliageGenerator.h"
#include "saiga/vulkan/world/PrototypeScene.h"
#include "saiga/vulkan/world/WorldCommon.h"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <memory>

#include "golden_image.h"
#include "render_scene_test_context.h"
#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace TerrainTestHooks;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

// A box of half extents with one colour (sRGB), visible to 1e6 m
Vulkan::MeshLods coloredBox(const vec3& half, const ucvec4& color)
{
    const std::array<std::pair<vec3, std::array<vec3, 4>>, 6> faces = {
        std::make_pair(vec3(0, 1, 0),
                       std::array<vec3, 4>{vec3(-1, 1, 1), vec3(1, 1, 1), vec3(1, 1, -1), vec3(-1, 1, -1)}),
        std::make_pair(vec3(0, -1, 0),
                       std::array<vec3, 4>{vec3(-1, -1, -1), vec3(1, -1, -1), vec3(1, -1, 1), vec3(-1, -1, 1)}),
        std::make_pair(vec3(1, 0, 0),
                       std::array<vec3, 4>{vec3(1, -1, 1), vec3(1, -1, -1), vec3(1, 1, -1), vec3(1, 1, 1)}),
        std::make_pair(vec3(-1, 0, 0),
                       std::array<vec3, 4>{vec3(-1, -1, -1), vec3(-1, -1, 1), vec3(-1, 1, 1), vec3(-1, 1, -1)}),
        std::make_pair(vec3(0, 0, 1),
                       std::array<vec3, 4>{vec3(-1, -1, 1), vec3(1, -1, 1), vec3(1, 1, 1), vec3(-1, 1, 1)}),
        std::make_pair(vec3(0, 0, -1),
                       std::array<vec3, 4>{vec3(1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1), vec3(1, 1, -1)})};
    Vulkan::MeshLod lod;
    lod.endDistance = 1e6f;
    for (const auto& f : faces)
    {
        const uint32_t base = uint32_t(lod.vertices.size());
        for (const vec3& c : f.second)
        {
            Vulkan::MeshVertex v;
            v.position = c.cwiseProduct(half);
            v.normal   = Vulkan::packNormalOct(f.first);
            v.color    = Vulkan::packColor(color);
            lod.vertices.push_back(v);
        }
        for (uint32_t i : {0u, 1u, 2u, 0u, 2u, 3u}) lod.indices.push_back(base + i);
    }
    Vulkan::MeshLods m;
    m.lods.push_back(lod);
    m.sphereRadius = half.norm();
    return m;
}

uint32_t hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
}  // namespace

// One scene, world and content for the suite (the scene's pipelines, the world's terrain and the trees are built once:
// test speed, AGENTS.md); each golden only sets the time, the weather, the exposure and the camera. Order independent:
// a shot completes its view's terrain, writes the environment and updates the environment map fully in its last frame
class LightingGolden : public ::testing::Test
{
   protected:
    static constexpr int size = 512;

    struct Suite
    {
        std::unique_ptr<Vulkan::HeadlessWindow> window;
        std::unique_ptr<Vulkan::RenderScene> scene;
        std::unique_ptr<World> world;
        std::unique_ptr<VulkanRenderSink> sink;
        Vulkan::PrototypeScene prototype;
        std::vector<Vulkan::InstanceSetHandle> sets;
        ~Suite()
        {
            if (scene)
            {
                scene->waitIdle();
                for (auto h : sets) scene->removeInstanceSet(h);
            }
            prototype.detach();
            if (world) world->setRenderSink(nullptr);
            sink.reset();
            world.reset();
            if (scene) scene->waitIdle();
            scene.reset();
            window.reset();
        }
    };
    static std::unique_ptr<Suite>& suite()
    {
        static std::unique_ptr<Suite> s;
        return s;
    }
    static void TearDownTestSuite() { suite().reset(); }

    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
        if (!suite()) CreateSuite();
        scene = suite()->scene.get();
        world = suite()->world.get();
        if (!scene->base().enabledFeatures.drawIndirectFirstInstance)
            GTEST_SKIP() << "drawIndirectFirstInstance is not supported";
    }

    static void CreateSuite()
    {
        auto st = std::make_unique<Suite>();
        WindowParameters windowParameters;
        windowParameters.width  = size;
        windowParameters.height = size;
        windowParameters.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
        Vulkan::VulkanParameters vulkanParameters;
        vulkanParameters.enableImgui = false;
        Vulkan::RenderSceneParameters p;
        p.imgui                    = false;
        p.msaaSamples              = 4;
        p.nearPlane                = 0.1f;
        p.toneMap                  = Vulkan::ToneMap::Aces;
        st->window                 = std::make_unique<Vulkan::HeadlessWindow>(windowParameters);
        st->scene                  = std::make_unique<Vulkan::RenderScene>(*st->window, vulkanParameters, p);
        Vulkan::RenderScene& scene = *st->scene;
        WorldSetup s;
        s.seed      = 1;
        s.profiling = false;
        s.config.spaceTemplates.push_back(testSpaceTemplate(16, 2, mountainRidgeHook, std::nullopt));
        s.generators          = registerTestHookGenerators;
        s.environment.enabled = true;
        st->world             = std::make_unique<World>(s);
        World& world          = *st->world;
        st->sink              = std::make_unique<VulkanRenderSink>(scene);
        world.setRenderSink(st->sink.get());

        // 200 trees west of the ridge (a hashed grid in [1100, 1700] x [1700, 2400]) and three boxes
        const auto species = Vulkan::FoliageGenerator::treeSpecies(1);
        std::vector<Vulkan::ForestRenderer::TreeInstance> trees;
        for (uint32_t i = 0; trees.size() < 200; ++i)
        {
            const uint32_t h = hash32(i * 2654435761u + 17u);
            const double x   = 1100 + double(h % 600);
            const double z   = 1700 + double((h >> 10) % 700);
            Vulkan::ForestRenderer::TreeInstance t;
            t.posYaw =
                vec4(float(x), float(world.terrainHeight(0, x, z) - 0.25), float(z), float((h >> 20) % 628) * 0.01f);
            t.scaleSpecies = vec4(8.0f + float((h >> 3) % 8), float(i % species.size()), 0, 0);
            trees.push_back(t);
        }
        st->prototype.settings.shading = Vulkan::WorldShading::Clay;
        st->prototype.attach(scene, trees, 1);
        const Vulkan::MaterialId material = Vulkan::MaterialId(scene.reserveIds(Vulkan::SceneIdKind::Material, 1));
        const Vulkan::MeshId mesh         = Vulkan::MeshId(scene.reserveIds(Vulkan::SceneIdKind::Mesh, 3));
        Vulkan::MaterialData metal;
        metal.roughness = 0.35f;
        metal.metalness = 0;
        scene.addMaterial(material, "golden_boxes", metal);
        scene.addMesh(mesh, "golden_red", coloredBox(vec3(6, 6, 6), ucvec4(190, 40, 30, 255)), material);
        scene.addMesh(Vulkan::MeshId(mesh + 1), "golden_white", coloredBox(vec3(4, 10, 4), ucvec4(230, 230, 225, 255)),
                      material);
        scene.addMesh(Vulkan::MeshId(mesh + 2), "golden_blue", coloredBox(vec3(8, 3, 5), ucvec4(40, 80, 190, 255)),
                      material);
        std::vector<Vulkan::StaticInstance> boxes;
        const std::array<vec3, 3> at = {vec3(1500, 0, 1900), vec3(1560, 0, 1960), vec3(1450, 0, 2000)};
        for (int k = 0; k < 3; ++k)
        {
            Vulkan::StaticInstance b;
            vec3 pos = at[size_t(k)];
            pos(1)   = float(world.terrainHeight(0, pos(0), pos(2))) + (k == 1 ? 10.0f : k == 0 ? 6.0f : 3.0f);
            b.setPosition(pos);
            b.scale = 1;
            b.mesh  = Vulkan::MeshId(mesh + k);
            boxes.push_back(b);
        }
        st->sets.push_back(scene.addInstanceSet(RegionId(), boxes));
        suite() = std::move(st);
    }

    TemplatedImage<ucvec4> RenderFrame()
    {
        TemplatedImage<ucvec4> img;
        scene->requestScreenshot([&](const TemplatedImage<ucvec4>& shot) { img = shot; });
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
        return img;
    }

    // The frame of a golden: the time of day and cloudiness written to the world, the fixed EV100, the environment of
    // the world's space 0 at the camera; the terrain of the view uploaded, then one frame with a full update of the
    // environment map (golden mode: the same image as forceFullUpdates in every frame, at one capture per shot)
    TemplatedImage<ucvec4> Shot(double hours, double cloudiness, float ev100, const Vec3& eye, const Vec3& forward,
                                double fovyDeg = 60)
    {
        std::string error;
        EXPECT_TRUE(setWorldTime(*world, WorldTimeChange{hours, std::nullopt, 0.0}, error)) << error;
        WeatherChange w;
        w.cloudiness = cloudiness;
        w.wind       = Vec2(0, 0);
        EXPECT_TRUE(setWeather(*world, 0, w, error)) << error;
        Vulkan::EnvironmentSettings settings;
        settings.exposure.mode     = Vulkan::ExposureMode::Fixed;
        settings.exposure.ev100    = ev100;
        Vulkan::EnvironmentState e = environmentState(*world, 0, 1.0, settings, world->environmentSetup());
        e.phases                   = {};
        scene->setEnvironment(e);
        const Quat rotation =
            Vulkan::viewRotation(lookAt(eye.cast<float>(), (eye + forward).cast<float>(), vec3(0, 1, 0)));
        EXPECT_TRUE(world->terrain().completeView(0, eye));
        scene->setCamera(0, eye, rotation, fovyDeg * kPi / 180);
        suite()->prototype.update();
        renderUntilUploaded(*scene, 256);
        scene->requestFullUpdate();
        TemplatedImage<ucvec4> img = RenderFrame();
        scene->waitIdle();
        return img;
    }
    double Ground(double x, double z) const { return world->terrainHeight(0, x, z); }
    std::string deviceName() const
    {
        return std::string(scene->base().physicalDevice.getProperties().deviceName.data());
    }

    Vulkan::RenderScene* scene = nullptr;
    World* world               = nullptr;
};

// Noon at 45 degrees latitude, a few clouds: the blue sky, the PBR surfaces in sunlight with the sky ambient, the cloud
// shadows on the ground; looking east at the ridge
TEST_F(LightingGolden, Noon)
{
    const Vec3 eye(1250, Ground(1250, 1900) + 30, 1900);
    auto img = Shot(12.0, 0.2, 15, eye, Vec3(1, 0.12, 0.45));
    EXPECT_EQ(img.w, size);
    compareGolden(img, "lighting_noon", deviceName());
}

// 17:45 at the equinox (the sun about 3 degrees above the western horizon; the plan named 18:40, after the sunset): the
// red sky towards the sun, the long light on the clouds and the trees against it; seen from west of the ridge looking
// west into the sun
TEST_F(LightingGolden, Sunset)
{
    const Vec3 eye(1300, Ground(1300, 2050) + 20, 2050);
    auto img              = Shot(17.75, 0.2, 11.5f, eye, Vec3(-1, 0.1, 0.12));
    const float elevation = float(std::asin(double(scene->environment().sunDir(1))) * 180 / kPi);
    EXPECT_GT(elevation, 0.5f);
    EXPECT_LT(elevation, 6.0f);
    compareGolden(img, "lighting_sunset", deviceName());
}

// 23:00 with the full moon in the south: the moon, the stars and the moonlit terrain
TEST_F(LightingGolden, Night)
{
    const Vec3 eye(1400, Ground(1400, 1900) + 30, 1900);
    auto img = Shot(23.0, 0.2, -1, eye, Vec3(0.2, 0.45, 1));
    EXPECT_GT(scene->environment().moonDir(1), 0.3f);
    compareGolden(img, "lighting_night", deviceName());
}

// Noon with 70% cloud cover seen from 2500 m above the layer at 2000 m: the cloud layer from above and the cloud
// shadows on the ground through its gaps (EV100 15.5: sunlit cloud tops are bright)
TEST_F(LightingGolden, Clouds)
{
    const Vec3 eye(1600, 2500, 1600);
    auto img = Shot(12.0, 0.7, 15.5f, eye, Vec3(0.4, -0.45, 0.6));
    EXPECT_TRUE(scene->lightingStatistics().clouds);
    compareGolden(img, "lighting_clouds", deviceName());
}
