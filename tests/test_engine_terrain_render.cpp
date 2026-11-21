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
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/World.h"
#include "saiga/vulkan/scene/GrassPass.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"

#include "gtest/gtest.h"

#include <cmath>
#include <iostream>
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
    const double y = yawDeg * M_PI / 180, p = pitchDeg * M_PI / 180;
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

// §22 acceptance (S20, A11): the 32 km space with a hook periodic in 4096 m (so in 28 672 m) and grass masks; two
// compositions (terrain + water looking 35 degrees down from 150 m; grass 30 degrees down from 2 m), each at pose A and
// at A + (28 672, 0, 28 672): the image pairs pass the golden metrics against each other
TEST_F(TerrainRenderTest, OriginOffsetWithinGoldenTolerance)
{
    Create(512, 512, 4);
    scene->setPrototypeEnvironment(Vulkan::PrototypeSettings(), 1);
    MakeWorld(testSpaceTemplate(128, 1, periodic4096Hook, 6 * 1024));
    const double fovy   = 40.0 * M_PI / 180;
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
