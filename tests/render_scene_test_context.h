/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/templatedImage.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/window/HeadlessWindow.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace Saiga
{
// The fixture of the RenderScene tests (S20): per test a HeadlessWindow and a RenderScene (imgui off, near plane 0.1;
// the VulkanTest fixture holds one static window with the forward renderer, while a RenderScene is its own renderer).
// Skipped without a Vulkan 1.3 device (as VulkanTest) and without drawIndirectFirstInstance (RenderScene needs it).
// TearDown waits for the device and destroys the scene before the window.
class RenderSceneTest : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
        Create(96, 64, 4);
        if (!scene->base().enabledFeatures.drawIndirectFirstInstance)
            GTEST_SKIP() << "drawIndirectFirstInstance is not supported";
    }
    void TearDown() override { Destroy(); }

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
        vulkanParameters.enableImgui = false;
        params.imgui                 = false;
        params.msaaSamples           = samples;
        params.nearPlane             = 0.1f;
        window                       = std::make_unique<Vulkan::HeadlessWindow>(windowParameters);
        scene                        = std::make_unique<Vulkan::RenderScene>(*window, vulkanParameters, params);
    }
    virtual void Destroy()
    {
        if (scene) scene->waitIdle();
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

    // Renders a frame and returns it
    TemplatedImage<ucvec4> RenderFrame()
    {
        TemplatedImage<ucvec4> img;
        scene->requestScreenshot([&](const TemplatedImage<ucvec4>& shot) { img = shot; });
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
        return img;
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
};

}  // namespace Saiga
