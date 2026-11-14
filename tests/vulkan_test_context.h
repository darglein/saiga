/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/templatedImage.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/window/HeadlessWindow.h"

#include "gtest/gtest.h"

#include <functional>

namespace Saiga
{
// The callbacks of the frames rendered by a test.
struct VulkanTestApp : public Vulkan::VulkanForwardRenderingInterface, public Updating
{
    std::function<void(vk::CommandBuffer)> onTransfer, onRender;
    int updates = 0;

    void transfer(vk::CommandBuffer cmd) override
    {
        if (onTransfer) onTransfer(cmd);
    }
    void render(vk::CommandBuffer cmd) override
    {
        if (onRender) onRender(cmd);
    }
    void update(float dt) override { updates++; }
};

// Base fixture of the Vulkan tests: a HeadlessWindow with the forward renderer for each test suite (no window system,
// no swap chain). Without a Vulkan 1.3 driver every test is skipped.
//
// The validation layer is enabled if it is installed. A validation error fails an assertion (see Vulkan::Debug),
// which aborts the test.
//
// Locally the best device is used (e.g. the NVIDIA GPU). To test on Mesa's software driver lavapipe, as CI does:
//   VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ctest -R test_vulkan
class VulkanTest : public ::testing::Test
{
   public:
    // Size of the render target
    static constexpr int width  = 64;
    static constexpr int height = 48;

    // The clear color of the forward renderer
    static ucvec4 ClearColor() { return ucvec4(57, 57, 57, 255); }

    static void SetUpTestSuite()
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) return;

        WindowParameters windowParameters;
        windowParameters.width  = width;
        windowParameters.height = height;
        // Needs SAIGA_WRITE_PROJECT_DIR_TO_HEADER, as the other tests that load files.
        windowParameters.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");

        Vulkan::VulkanParameters vulkanParameters;
        vulkanParameters.enableImgui = false;

        window   = std::make_unique<Vulkan::HeadlessWindow>(windowParameters);
        renderer = std::make_unique<Vulkan::VulkanForwardRenderer>(*window, vulkanParameters);
        renderer->setRenderObject(app);
        window->setUpdateObject(app);
    }

    static void TearDownTestSuite()
    {
        renderer.reset();
        window.reset();
    }

   protected:
    void SetUp() override
    {
        if (!renderer) GTEST_SKIP() << "No Vulkan 1.3 device available";
        app = VulkanTestApp();
    }

    void TearDown() override
    {
        // The objects of the test are destroyed after this.
        if (renderer) renderer->waitIdle();
        app = VulkanTestApp();
    }

    static Vulkan::VulkanBase& base() { return renderer->base(); }

    // Renders one frame and returns it. Row 0 is the top row.
    static TemplatedImage<ucvec4> RenderFrame()
    {
        TemplatedImage<ucvec4> result;
        renderer->requestScreenshot([&](const TemplatedImage<ucvec4>& img) { result = img; });
        window->renderFrames(1);
        return result;
    }

    // Records the commands with the function, submits them to the main queue and waits.
    static void Submit(const std::function<void(vk::CommandBuffer)>& record)
    {
        auto& queue           = base().mainQueue;
        vk::CommandBuffer cmd = queue.commandPool.createAndBeginOneTimeBuffer();
        record(cmd);
        cmd.end();
        queue.submitAndWait(cmd);
        queue.commandPool.freeCommandBuffer(cmd);
    }

    static inline VulkanTestApp app;
    static inline std::unique_ptr<Vulkan::HeadlessWindow> window;
    static inline std::unique_ptr<Vulkan::VulkanForwardRenderer> renderer;
};

}  // namespace Saiga
