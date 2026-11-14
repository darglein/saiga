/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/window/Window.h"

namespace Saiga
{
namespace Vulkan
{
/**
 * A window without a window system: no surface, no swap chain, no input and no GUI. A renderer created with it
 * (VulkanRenderer::isHeadless) renders into offscreen images of the window size, which are read with
 * VulkanForwardRenderer::requestScreenshot.
 *
 * This runs without a display (CI, servers) and on software drivers such as Mesa's lavapipe.
 * Check Instance::suitableDeviceAvailable() first if a Vulkan driver may be missing.
 *
 *   HeadlessWindow window(windowParameters);
 *   VulkanForwardRenderer renderer(window, vulkanParameters);
 *   renderer.setRenderObject(app);
 *   renderer.requestScreenshot([](const TemplatedImage<ucvec4>& img) { ... });
 *   window.renderFrames(1);
 */
class SAIGA_VULKAN_API HeadlessWindow : public VulkanWindow
{
   public:
    HeadlessWindow(WindowParameters windowParameters);
    ~HeadlessWindow() override;

    std::vector<std::string> getRequiredInstanceExtensions() override { return {}; }

    // Creates no surface, which makes the renderer headless.
    void createSurface(VkInstance instance, VkSurfaceKHR* surface) override;

    void update(float dt) override;

    /**
     * Updates and renders the given number of frames as fast as possible, without the timing of the main loop
     * (startMainLoop also works, it runs until close() is called).
     */
    void renderFrames(int count, float dt = 1.0f / 60.0f);
};

}  // namespace Vulkan
}  // namespace Saiga
