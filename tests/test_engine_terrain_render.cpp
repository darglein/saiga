/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine's terrain through the render sink on a RenderScene (docs/engine_design.md §12.2, §17.6; S20): a World
// with a test space template streams into the scene (TerrainSystem::completeView, a VulkanRenderSink); the origin
// offset acceptance of §22.

#include "saiga/engine/core/Log.h"
#include "saiga/engine/sync/VulkanRenderSink.h"
#include "saiga/engine/terrain/TerrainEdit.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainStore.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/World.h"
#include "saiga/vulkan/scene/GrassPass.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/world/PrototypeScene.h"
#include "saiga/vulkan/world/PrototypeTerrain.h"
#include "saiga/vulkan/world/WorldCommon.h"

#include "gtest/gtest.h"

#include <cmath>
#include <iostream>
#include <memory>

#include "forest_reference.h"
#include "golden_image.h"
#include "render_scene_test_context.h"
#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace TerrainTestHooks;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

// The camera rotation looking along the direction (y up)
Quat lookRotation(const Vec3& forward)
{
    const Vec3 f = forward.normalized();
    const Vec3 r = f.cross(Vec3(0, 1, 0)).normalized();
    const Vec3 u = r.cross(f);
    Mat3 m;
    for (int i = 0; i < 3; ++i)
    {
        m(i, 0) = r(i);
        m(i, 1) = u(i);
        m(i, 2) = -f(i);
    }
    return Quat(m).normalized();
}
// Along yaw (degrees from +x towards +z) and pitch (degrees, negative: down)
Vec3 direction(double yawDeg, double pitchDeg)
{
    const double y = yawDeg * kPi / 180, p = pitchDeg * kPi / 180;
    return Vec3(std::cos(p) * std::cos(y), std::sin(p), std::cos(p) * std::sin(y));
}
}  // namespace

// RenderSceneTest with a World whose sink is a VulkanRenderSink on the scene (constructed after the scene, destroyed
// before it); captures complete the view first (completeView) and wait for the uploads
class TerrainRenderTest : public RenderSceneTest
{
   protected:
    void MakeWorld(const SpaceTemplate& space, uint64_t seed = 1)
    {
        DestroyWorld();
        WorldSetup s;
        s.seed      = seed;
        s.profiling = false;
        s.config.spaceTemplates.push_back(space);
        world = std::make_unique<World>(s);
        sink  = std::make_unique<VulkanRenderSink>(*scene);
        world->setRenderSink(sink.get());
    }
    void DestroyWorld()
    {
        if (world) world->setRenderSink(nullptr);
        sink.reset();
        world.reset();
    }
    void Destroy() override
    {
        DestroyWorld();
        RenderSceneTest::Destroy();
    }
    TemplatedImage<ucvec4> CaptureView(const Vec3& eye, const Quat& rotation, double fovy, int frames = 3)
    {
        EXPECT_TRUE(world->terrain().completeView(0, eye));
        SetCamera(0, eye, rotation, fovy);
        renderUntilUploaded(*scene, 256);
        return Capture(frames);
    }

    std::unique_ptr<World> world;
    std::unique_ptr<VulkanRenderSink> sink;
};

// S22: an engine-path check of a terrain patch below the tile's minimum: after the dirty queue drained, the atlas
// layer holds the merged heights, the pit's level-0 node is selected (not culled by the old bounds) and the probe at
// the pit's centre returns the patched height
TEST_F(TerrainRenderTest, PatchBelowNodeMinimum)
{
    Create(256, 256, 1);
    MakeWorld(terraformTestTemplate(2));
    const Vec3 pit(300, 0, 300);
    const double ground = world->terrainHeight(0, pit.x(), pit.z());
    const Vec3 eye(pit.x(), ground + 60, pit.z() - 50);
    const Quat rot = lookRotation(Vec3(pit.x(), ground, pit.z()) - eye);
    CaptureView(eye, rot, 1.0);
    RegionId tile;
    tile.level                  = 0;
    tile.x                      = 1;
    tile.z                      = 1;
    const TerrainTileData* sent = nullptr;
    for (const auto& [r, t] : world->terrain().store(0).tiles())
        if (r == tile) sent = &t.data;
    ASSERT_TRUE(sent);
    // 30 m below the tile's minimum at the centre sample (150, 150)
    const int32_t center = world->terrainSample(0, 150, 150);
    const double depth   = double(center - sent->minHeight) * kHeightUnit + 30;
    TerrainBrush b;
    b.op       = TerrainBrush::Subtract;
    b.center   = Vec3(pit.x(), 0, pit.z());
    b.radius   = 20;
    b.strength = std::ceil(depth);
    applyTerrainEdit(*world, 0, b);
    const int32_t bottom = world->terrainSample(0, 150, 150);
    ASSERT_LT(bottom, sent->minHeight - 30 * 1024);
    for (int i = 0;
         i < 64 && (world->terrain().dirtyQueue().size() > 0 || world->terrain().terrainUploadsPending() > 0); ++i)
    {
        world->terrain().updateView(0, eye);
        Capture(1);
    }
    EXPECT_EQ(world->terrain().dirtyQueue().size(), 0u);
    EXPECT_EQ(world->terrain().terrainUploadsPending(), 0u);
    // The layer equals the merged heights
    std::vector<int32_t> expected(129 * 129);
    for (int k = 0; k < 129; ++k)
        for (int i = 0; i < 129; ++i) expected[size_t(k * 129 + i)] = world->terrainSample(0, 128 + i, 128 + k);
    EXPECT_EQ(scene->terrainAtlas().readLayerForTests(tile), expected);
    // The pit's level-0 node is selected and its probe returns the patched height
    Capture(2);
    const auto& nodes = scene->terrainPass().selectedNodes();
    int found         = -1;
    for (size_t n = 0; n < nodes.size(); ++n)
        if (nodes[n].level == 0 && !nodes[n].quarter && !nodes[n].water && nodes[n].originX <= 300 &&
            nodes[n].originX + 64 >= 300 && nodes[n].originZ <= 300 && nodes[n].originZ + 64 >= 300)
            found = int(n);
    ASSERT_GE(found, 0) << "the pit's level-0 node is not selected";
    const int gx   = (300 - nodes[size_t(found)].originX) / 2;
    const int gz   = (300 - nodes[size_t(found)].originZ) / 2;
    const auto out = scene->terrainPass().probeVertices({Vulkan::TerrainProbe{found, gx, gz}});
    const Vec3 v   = eye + Vec3(out[0](0), out[0](1), out[0](2));
    const double d = (Vec3(300, double(bottom) * kHeightUnit, 300) - eye).norm();
    EXPECT_NEAR(v.x(), 300, 1e-6 * d + 1e-4);
    EXPECT_NEAR(v.z(), 300, 1e-6 * d + 1e-4);
    EXPECT_NEAR(v.y(), double(bottom) * kHeightUnit, 1e-6 * d + 1e-4);
}

// S22: the golden of a terraformed crater: a crater applied before the first completeView (the merge path) and a ring
// of mounds applied after it (the patch path, the edited grass masks); 512 x 512 on lavapipe
TEST_F(TerrainRenderTest, EditCraterGolden)
{
    Create(512, 512, 4);
    scene->setPrototypeEnvironment(Vulkan::PrototypeSettings(), 1);
    MakeWorld(terraformTestTemplate(2));
    auto edit = [&](TerrainBrush::Op op, double x, double z, double radius, double strength)
    {
        TerrainBrush b;
        b.op       = op;
        b.center   = Vec3(x, 0, z);
        b.radius   = radius;
        b.strength = strength;
        applyTerrainEdit(*world, 0, b);
    };
    const Vec3 c(300, 0, 300);
    edit(TerrainBrush::Subtract, c.x(), c.z(), 20, 6);
    const double ground = world->terrainHeight(0, c.x() - 60, c.z() - 60);
    const Vec3 eye(c.x() - 45, ground + 22, c.z() - 45);
    const Quat rot    = lookRotation(Vec3(c.x(), ground - 4, c.z()) - eye);
    const double fovy = 60.0 * M_PI / 180;
    CaptureView(eye, rot, fovy);
    // The ring: 8 mounds of 2 m on a 24 m circle (integer offsets)
    const int ring[8][2] = {{24, 0}, {17, 17}, {0, 24}, {-17, 17}, {-24, 0}, {-17, -17}, {0, -24}, {17, -17}};
    for (const auto& r : ring) edit(TerrainBrush::Add, c.x() + r[0], c.z() + r[1], 6, 2);
    for (int i = 0; i < 200; ++i)
    {
        world->terrain().updateView(0, eye);
        Capture(1);
        const TerrainSystem::Stats& s = world->terrain().stats();
        if (world->terrain().dirtyQueue().size() == 0 && s.maskQueue == 0 && s.maskJobsInFlight == 0 &&
            world->terrain().terrainUploadsPending() == 0)
            break;
    }
    EXPECT_EQ(world->terrain().dirtyQueue().size(), 0u);
    EXPECT_EQ(world->terrain().stats().maskJobsInFlight, 0);
    EXPECT_GT(world->terrain().stats().maskRecomputes, 0u);
    SetCamera(0, eye, rot, fovy);
    const auto img = Capture(3);
    compareGolden(img, "terrain_edit_crater", deviceName());
}

// §22 acceptance (S20, A11): the 32 km space with a hook periodic in 4096 m (so in 28 672 m) and grass masks; two
// compositions (terrain + water looking 35 degrees down from 150 m; grass 30 degrees down from 2 m), each at pose A and
// at A + (28 672, 0, 28 672): the image pairs pass the golden metrics against each other
TEST_F(TerrainRenderTest, OriginOffsetWithinGoldenTolerance)
{
    Create(512, 512, 4);
    scene->setPrototypeEnvironment(Vulkan::PrototypeSettings(), 1);
    MakeWorld(testSpaceTemplate(128, 1, periodic4096Hook, 6 * 1024));
    const double fovy   = 40.0 * kPi / 180;
    const double offset = 28672;
    struct Composition
    {
        const char* name;
        Vec3 xz;
        double above, pitch;
    };
    for (const Composition& c :
         {Composition{"terrain", Vec3(1500, 0, 1500), 150, -35}, Composition{"grass", Vec3(1700, 0, 1500), 2, -30}})
    {
        std::vector<TemplatedImage<ucvec4>> images;
        std::vector<int> nodes, blocks;
        for (int k = 0; k < 2; ++k)
        {
            const Vec3 p   = c.xz + Vec3(offset * k, 0, offset * k);
            const double h = std::max(world->terrainHeight(0, p.x(), p.z()), 6.0);
            images.push_back(CaptureView(p + Vec3(0, h + c.above, 0), lookRotation(direction(0, c.pitch)), fovy));
            nodes.push_back(scene->terrainPassStats().nodes);
            if (c.above > 100)
            {
                EXPECT_TRUE(scene->terrainPassStats().water);
            }
            blocks.push_back(scene->grassPass().stats().blocks);
        }
        const GoldenMetrics m = goldenMetrics(images[0], images[1]);
        std::cout << "OriginOffset " << c.name << ": mean abs " << m.meanAbs << "/255, " << m.largeFraction * 100
                  << "% > 16/255, max " << m.maxDelta << ", nodes " << nodes[0] << " / " << nodes[1]
                  << ", grass blocks " << blocks[0] << " / " << blocks[1] << std::endl;
        EXPECT_TRUE(m.pass) << c.name;
        EXPECT_EQ(nodes[0], nodes[1]) << c.name;
        EXPECT_EQ(blocks[0], blocks[1]) << c.name;
        if (std::string(c.name) == "grass")
        {
            EXPECT_GT(blocks[0], 0);
        }
    }
}

// Golden images of the prototype world (docs/engine_design.md §17.6; S20: moved from test_vulkan_world): 512 x 512,
// clay shading, time 0, captured after 5 frames. The engine's terrain: the prototype hook at base spacing 2 (CI cost),
// a 128-chunk space with the sea level 6 m, completeView and the upload wait before the capture; 5000 trees placed in
// the 1 km square [1024, 2048]^2 (seed 1) on PrototypeScene. They gate on lavapipe only; other devices print the
// metrics. The drawn instance and terrain node counts gate on lavapipe only as well (the tree placement uses std random
// distributions and float comparisons); the counts are also compared with the CPU reference of the culling rule (§17.5
// band). Regenerate on lavapipe: SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_terrain_render
// --gtest_filter='VulkanWorldGolden.*'
class VulkanWorldGolden : public TerrainRenderTest
{
   protected:
    static constexpr int size = 512;

    void SetUp() override
    {
        TerrainRenderTest::SetUp();
        if (IsSkipped()) return;
        Create(size, size, 4);
        MakeWorld(
            testSpaceTemplate(Vulkan::kPrototypeSpaceChunks, 2, Vulkan::prototypeHeight, Vulkan::kPrototypeSeaLevel));
        Vulkan::PrototypeArea area;
        area.x0                    = 1024;
        area.z0                    = 1024;
        area.size                  = 1024;
        area.seed                  = world->spaceSeed(0);
        area.baseSpacing           = 2;
        species                    = Vulkan::FoliageGenerator::treeSpecies(1);
        trees                      = Vulkan::ForestRenderer::placeTrees(area, 5000, 1, species);
        prototype.settings.shading = Vulkan::WorldShading::Clay;
        prototype.attach(*scene, trees, 1);
        scene->setTime(0);
    }
    void Destroy() override
    {
        prototype.detach();
        TerrainRenderTest::Destroy();
    }
    double Ground(double x, double z) const { return world->terrainHeight(0, x, z); }
    TemplatedImage<ucvec4> Shot(const Vec3& eye, const Vec3& target)
    {
        const Quat rotation = Vulkan::viewRotation(lookAt(eye.cast<float>(), target.cast<float>(), vec3(0, 1, 0)));
        EXPECT_TRUE(world->terrain().completeView(0, eye));
        SetCamera(0, eye, rotation, double(radians(prototype.settings.fovy)));
        prototype.update();
        renderUntilUploaded(*scene, 256);
        return Capture(5);
    }

    // The committed counts (docs/engine_design.md §17.6)
    void ExpectCounts(const char* name, int lod0, int lod1, int lod2, int terrainNodes)
    {
        const auto fs  = prototype.forest().stats();
        const auto& st = scene->statistics();
        const auto ref = ForestReferenceTests::forestReference(trees, species, prototype.settings, cameraEye,
                                                               cameraRotation, double(width) / height);
        std::cout << name << " trees per lod: " << fs.visible[0] << " " << fs.visible[1] << " " << fs.visible[2]
                  << ", terrain nodes " << st.terrainNodes << " (reference " << ref.perLod[0] << " " << ref.perLod[1]
                  << " " << ref.perLod[2] << ", " << ref.ambiguous << " ambiguous)" << std::endl;
        EXPECT_EQ(fs.frame, scene->frames());
        ForestReferenceTests::expectForest(fs, ref, name);
        // Not checked while the goldens are regenerated: the printed counts replace the committed ones
        if (!goldenGating(deviceName()) || goldenUpdateMode()) return;
        EXPECT_EQ(int(fs.visible[0]), lod0);
        EXPECT_EQ(int(fs.visible[1]), lod1);
        EXPECT_EQ(int(fs.visible[2]), lod2);
        EXPECT_EQ(int(st.terrainNodes), terrainNodes);
    }

    Vulkan::PrototypeScene prototype;
    std::vector<Vulkan::TreeSpecies> species;
    std::vector<Vulkan::ForestRenderer::TreeInstance> trees;
};

TEST_F(VulkanWorldGolden, WorldOverview)
{
    // 300 m above the ground 100 m inside the square's west edge at its centre z, looking 45 degrees down along +x
    Vec3 c(1124, 0, 1536);
    c(1)           = Ground(c(0), c(2));
    const Vec3 eye = c + Vec3(0, 300, 0);
    const Vec3 dir(std::cos(kPi / 4), -std::sin(kPi / 4), 0);
    auto img = Shot(eye, eye + dir);
    EXPECT_EQ(img.w, 512);
    compareGolden(img, "world_overview", deviceName());
    ExpectCounts("world_overview", 0, 2, 3937, 65);
}

TEST_F(VulkanWorldGolden, WorldForest)
{
    // 2 m above max(ground, water) at a clearing near the square's centre: (1524, 1564), found by a search on a 4 m
    // grid of [1400, 1700]^2 for the point nearest to the centre whose nearest tree is >= 15 m away (16.2 m), looking
    // along +x and slightly down: a close tree with alpha-to-coverage foliage, grass, trunks, the forest at mid
    // distance
    const Vec3 best(1524, 0, 1564);
    Vec3 eye = best;
    eye(1)   = std::max(Ground(eye(0), eye(2)), 6.0) + 2.0;
    auto img = Shot(eye, eye + Vec3(1, -0.2, 0));
    EXPECT_EQ(img.w, 512);
    compareGolden(img, "world_forest", deviceName());
    ExpectCounts("world_forest", 27, 338, 506, 131);
}
