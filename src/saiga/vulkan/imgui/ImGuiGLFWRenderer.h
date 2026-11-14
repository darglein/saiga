/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"

struct GLFWwindow;

namespace Saiga
{
namespace Vulkan
{
/**
 * ImGui for a GLFW window: input and window handling by the ImGui GLFW backend (saiga/core/glfw/imgui_impl_glfw.h),
 * rendering by the ImGuiVulkanRenderer.
 *
 * The Vulkan renderer draws only into the main window, so ImGui's multi-viewport support is disabled.
 */
class SAIGA_VULKAN_API ImGuiGLFWRenderer : public ImGuiVulkanRenderer
{
   public:
    ImGuiGLFWRenderer(size_t frameCount, const ImGuiParameters& params, GLFWwindow* window);
    ~ImGuiGLFWRenderer() override;

    void beginFrame() override;
};

}  // namespace Vulkan
}  // namespace Saiga
