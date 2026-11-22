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

// The fixture of the lighting tests (S36a1b, finding 12 of the plan's review 1): one HeadlessWindow and RenderScene per
// test suite (the scene shaders are compiled per RenderScene: a scene per test would dominate the binary's run time),
// 128 x 128 with 4 samples by default (a derived fixture with other values is its own suite: override Size / Samples).
// SetUp resets the state through the public calls: the lighting parameters, an environment at its defaults, Identity,
// exposure 1, no studio override, the sky ambient, the terrain cleared and the instance sets added with AddSet removed.
// Materials and meshes stay registered: each test uses its own IDs. Skipped as RenderSceneTest.
class RenderSceneSuiteTest : public ::testing::Test
{
   protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite()
    {
        if (suiteScene()) suiteScene()->waitIdle();
        suiteScene().reset();
        suiteWindow().reset();
    }

    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
        if (!suiteScene()) CreateSuite(Size(), Samples());
        scene = suiteScene().get();
        if (!scene->base().enabledFeatures.drawIndirectFirstInstance)
            GTEST_SKIP() << "drawIndirectFirstInstance is not supported";
        width  = Size()(0);
        height = Size()(1);
        scene->setLightingParameters(Vulkan::LightingParameters());
        scene->setEnvironment(Vulkan::EnvironmentState());
        scene->setToneMap(Vulkan::ToneMap::Identity);
        scene->setExposure(1);
        scene->setStudioEnvironment(std::nullopt);
        scene->skyAmbient().setUniformRadiance(std::nullopt);
        scene->clearTerrain();
        for (auto h : sets) scene->removeInstanceSet(h);
        sets.clear();
        // The debug state of a case that failed midway: the view, the heat map, a running probe sequence (its frames)
        scene->setLightingDebug(Vulkan::LightingDebugState());
        scene->setLightingDebugOverride(std::nullopt);
        scene->setExposureViewRange(std::nullopt, Vulkan::RenderScene::kDefaultExposureViewRange);
        for (int i = 0; i < 2 * Vulkan::RenderScene::kProbeFrames && scene->probePending(); ++i) RenderFrame();
        ASSERT_FALSE(scene->probePending());
    }
    void TearDown() override
    {
        if (scene)
        {
            for (auto h : sets) scene->removeInstanceSet(h);
            sets.clear();
        }
    }

    // The suite's size and samples (a derived fixture's suite may override them)
    virtual ivec2 Size() const { return ivec2(128, 128); }
    virtual int Samples() const { return 4; }

    Vulkan::InstanceSetHandle AddSet(const std::vector<Vulkan::StaticInstance>& instances)
    {
        sets.push_back(scene->addInstanceSet(RegionId(), instances));
        return sets.back();
    }
    void SetCamera(SpaceId space, const Vec3& eye, const Quat& rotation, double fovy)
    {
        scene->setCamera(space, eye, rotation, fovy);
    }
    TemplatedImage<ucvec4> RenderFrame()
    {
        TemplatedImage<ucvec4> img;
        scene->requestScreenshot([&](const TemplatedImage<ucvec4>& shot) { img = shot; });
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
        return img;
    }
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
    Vulkan::RenderScene* scene = nullptr;
    std::vector<Vulkan::InstanceSetHandle> sets;

    static std::unique_ptr<Vulkan::HeadlessWindow>& suiteWindow()
    {
        static std::unique_ptr<Vulkan::HeadlessWindow> w;
        return w;
    }
    static std::unique_ptr<Vulkan::RenderScene>& suiteScene()
    {
        static std::unique_ptr<Vulkan::RenderScene> s;
        return s;
    }

   private:
    static void CreateSuite(ivec2 size, int samples)
    {
        WindowParameters windowParameters;
        windowParameters.width  = size(0);
        windowParameters.height = size(1);
        windowParameters.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
        Vulkan::VulkanParameters vulkanParameters;
        vulkanParameters.enableImgui = false;
        Vulkan::RenderSceneParameters params;
        params.imgui       = false;
        params.msaaSamples = samples;
        params.nearPlane   = 0.1f;
        suiteWindow()      = std::make_unique<Vulkan::HeadlessWindow>(windowParameters);
        suiteScene()       = std::make_unique<Vulkan::RenderScene>(*suiteWindow(), vulkanParameters, params);
    }
};

}  // namespace Saiga
