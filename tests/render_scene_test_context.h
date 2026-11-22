/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/templatedImage.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/SkyAmbient.h"
#include "saiga/vulkan/window/HeadlessWindow.h"

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Saiga
{
// The lighting of the goldens (S36a2, §17.6): golden mode (forceFullUpdates: every incremental effect complete in
// every frame) with the cascades at 1024 and the horizon map at 256 texels (lavapipe's cost; recorded here for every
// golden of the scene)
inline Vulkan::RenderSceneParameters goldenSceneParameters(Vulkan::RenderSceneParameters p = {})
{
    p.lighting.forceFullUpdates  = true;
    p.lighting.cascadeResolution = 1024;
    p.lighting.horizonResolution = 256;
    return p;
}

// The fixture of the RenderScene tests (S20): per test a HeadlessWindow and a RenderScene (imgui off, near plane 0.1,
// S36a2: golden mode, LightingParameters::forceFullUpdates: the incremental effects (the horizon map, the staggered
// cascades, the environment map) are complete in every frame, so frames compare; the VulkanTest fixture holds one
// static window with the forward renderer, while a RenderScene is its own renderer).
// Size() x Samples() (96 x 64, 4 samples) unless a derived fixture overrides them. A new scene per test is cheap with
// the device and pipeline reuse of the test process (AGENTS.md, Tests, "Test speed"; about 15 ms on lavapipe).
// Skipped without a Vulkan 1.3 device (as VulkanTest) and without drawIndirectFirstInstance (RenderScene needs it).
// TearDown waits for the device and destroys the scene before the window.
class RenderSceneTest : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
        Create(Size()(0), Size()(1), Samples());
        if (!scene->base().enabledFeatures.drawIndirectFirstInstance)
            GTEST_SKIP() << "drawIndirectFirstInstance is not supported";
    }
    void TearDown() override { Destroy(); }

    // The scene's size and samples (a derived fixture may override them)
    virtual ivec2 Size() const { return ivec2(96, 64); }
    virtual int Samples() const { return 4; }

    void Create(int w, int h, int samples, Vulkan::RenderSceneParameters params = Vulkan::RenderSceneParameters())
    {
        Destroy();
        width  = w;
        height = h;
        WindowParameters windowParameters;
        windowParameters.width  = w;
        windowParameters.height = h;
        windowParameters.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
        Vulkan::VulkanParameters vulkanParameters;
        vulkanParameters.enableImgui     = false;
        params.imgui                     = false;
        params.msaaSamples               = samples;
        params.nearPlane                 = 0.1f;
        params.lighting.forceFullUpdates = true;
        window                           = std::make_unique<Vulkan::HeadlessWindow>(windowParameters);
        scene                            = std::make_unique<Vulkan::RenderScene>(*window, vulkanParameters, params);
    }
    virtual void Destroy()
    {
        if (scene) scene->waitIdle();
        sets.clear();
        scene.reset();
        window.reset();
    }

    // The camera: space, eye, rotation (looks along -z of it), vertical field of view in radians
    void SetCamera(SpaceId space, const Vec3& eye, const Quat& rotation, double fovy)
    {
        cameraSpace    = space;
        cameraEye      = eye;
        cameraRotation = rotation;
        cameraFovy     = fovy;
        scene->setCamera(space, eye, rotation, fovy);
    }

    // A static instance set in region 0 (removed with the scene)
    Vulkan::InstanceSetHandle AddSet(const std::vector<Vulkan::StaticInstance>& instances)
    {
        sets.push_back(scene->addInstanceSet(RegionId(), instances));
        return sets.back();
    }

    // Renders a frame and returns it. The callback owns its image: a probe sequence defers a screenshot into a later
    // frame (the image is then empty here)
    TemplatedImage<ucvec4> RenderFrame()
    {
        auto img = std::make_shared<TemplatedImage<ucvec4>>();
        scene->requestScreenshot([img](const TemplatedImage<ucvec4>& shot) { *img = shot; });
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
        return *img;
    }

    // Renders n frames and returns the last; waits for it (its statistics are the newest)
    TemplatedImage<ucvec4> Capture(int n = 3)
    {
        TemplatedImage<ucvec4> img;
        for (int i = 0; i < n; ++i) img = RenderFrame();
        scene->waitIdle();
        return img;
    }

    std::string deviceName() const
    {
        return std::string(scene->base().physicalDevice.getProperties().deviceName.data());
    }

    int width = 0, height = 0;
    SpaceId cameraSpace = 0;
    Vec3 cameraEye      = Vec3::Zero();
    Quat cameraRotation = Quat::Identity();
    double cameraFovy   = 1;
    std::unique_ptr<Vulkan::HeadlessWindow> window;
    std::unique_ptr<Vulkan::RenderScene> scene;
    std::vector<Vulkan::InstanceSetHandle> sets;
};

}  // namespace Saiga
