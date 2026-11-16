/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Headless test of the transitional mesh renderer of the tech demo games (games/common/MeshRenderer.h).

#include "games/common/MeshRenderer.h"
#include "gtest/gtest.h"

#include "vulkan_test_context.h"

using namespace Saiga;
using namespace Saiga::Game;

namespace
{
// The configuration of the games: MSAA 4 and reversed depth (VulkanTest uses the default renderer)
class GameMeshRenderer : public ::testing::Test
{
   protected:
    static constexpr int width = 96, height = 64;

    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
        WindowParameters windowParameters;
        windowParameters.width  = width;
        windowParameters.height = height;
        windowParameters.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
        Vulkan::VulkanParameters vulkanParameters;
        vulkanParameters.enableImgui = false;
        Vulkan::ForwardRenderingParameters params;
        params.msaaSamples = 4;
        params.reversedZ   = true;
        window             = std::make_unique<Vulkan::HeadlessWindow>(windowParameters);
        renderer           = std::make_unique<Vulkan::VulkanForwardRenderer>(*window, vulkanParameters, params);
        renderer->setRenderObject(app);
        window->setUpdateObject(app);
    }
    void TearDown() override
    {
        if (renderer) renderer->waitIdle();
        app = VulkanTestApp();
        meshes.destroy();
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
    MeshRenderer meshes;
};
}  // namespace

TEST_F(GameMeshRenderer, NearBoxOccludesFarBoxFarFromTheOrigin)
{
    meshes.init(*renderer);
    int red   = meshes.addMesh(MeshBuilder::box(Vec3(1, 1, 1), Vec3(0.9, 0.1, 0.1)));
    int green = meshes.addMesh(MeshBuilder::box(Vec3(6, 6, 1), Vec3(0.1, 0.9, 0.1)));

    // Far from the origin (camera relative rendering), looking along -z: a small red box 6 m ahead in front of a large
    // green box 20 m ahead
    Vec3 eye(7000.25, 300.5, -4000.75);
    mat4 view = lookAt(eye.cast<float>(), (eye + Vec3(0, 0, -1)).cast<float>(), vec3(0, 1, 0));
    mat4 proj = Vulkan::reversedZInfinitePerspective(radians(60.0f), float(width) / height, 0.1f);
    MeshRenderer::Lighting light;
    light.sunDirection = Vec3(0, 0, 1);  // towards the camera: the front faces are lit

    app.onTransfer = [&](vk::CommandBuffer cmd) { meshes.transfer(cmd, view, eye, proj, light); };
    app.onRender   = [&](vk::CommandBuffer cmd)
    {
        // The far box first: with a wrong depth test (not reversed) it would cover the near box
        meshes.draw(green, transformMatrix(eye + Vec3(0, 0, -20), Quat::Identity()));
        meshes.draw(red, transformMatrix(eye + Vec3(0, 0, -6), Quat::Identity()));
        meshes.render(cmd);
    };
    TemplatedImage<ucvec4> img = RenderFrame();
    ASSERT_EQ(img.w, width);
    renderer->waitIdle();

    auto isRed   = [](ucvec4 c) { return c(0) > 2 * c(1) && c(0) > 2 * c(2) && c(0) > 60; };
    auto isGreen = [](ucvec4 c) { return c(1) > 2 * c(0) && c(1) > 2 * c(2) && c(1) > 60; };
    // The red box covers about +-9 pixels around the center, the green one about +-16
    ucvec4 center = img(height / 2, width / 2), side = img(height / 2, width / 2 + 13), corner = img(1, 1);
    EXPECT_TRUE(isRed(center)) << int(center(0)) << " " << int(center(1)) << " " << int(center(2));
    EXPECT_TRUE(isGreen(side)) << int(side(0)) << " " << int(side(1)) << " " << int(side(2));
    EXPECT_FALSE(isRed(corner) || isGreen(corner));
    EXPECT_GT(meshes.drawnTriangles(), 0);
}
