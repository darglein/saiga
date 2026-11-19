/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui_headless_platform.h"
#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"

namespace Saiga
{
class WindowBase;

namespace Vulkan
{
/**
 * ImGui on a window without a window system (HeadlessWindow): drawn by the renderer into its offscreen images, see
 * ImGuiHeadlessPlatform for input, viewports and the clipboard.
 */
class SAIGA_VULKAN_API ImGuiHeadlessRenderer : public ImGuiVulkanRenderer
{
   public:
    ImGuiHeadlessRenderer(size_t frameCount, const ImGuiParameters& params, WindowBase& window);

    void beginFrame() override;

   private:
    WindowBase& window;
    ImGuiHeadlessPlatform platform;
};

}  // namespace Vulkan
}  // namespace Saiga
