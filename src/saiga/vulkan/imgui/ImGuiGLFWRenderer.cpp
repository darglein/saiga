/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_GLFW
#    include "saiga/core/glfw/imgui_impl_glfw.h"
#    include "saiga/core/imgui/imgui.h"

#    include "ImGuiGLFWRenderer.h"

#    if defined(SAIGA_OPENGL_INCLUDED)
#        error OpenGL was included somewhere.
#    endif

namespace Saiga
{
namespace Vulkan
{
ImGuiGLFWRenderer::ImGuiGLFWRenderer(size_t frameCount, const ImGuiParameters& params, GLFWwindow* window)
    : ImGuiVulkanRenderer(frameCount, params)
{
    // initImGui() enables multi-viewports, which need a renderer backend for the platform windows. The Vulkan renderer
    // only draws into the main window. This must be cleared before the GLFW backend is initialized.
    ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;

    // Installs the ImGui callbacks and chains them to the ones already set on the window (glfw_EventHandler).
    bool ok = ImGui_ImplGlfw_InitForVulkan(window, true);
    SAIGA_ASSERT(ok);
}

ImGuiGLFWRenderer::~ImGuiGLFWRenderer()
{
    // Before ~ImGuiRenderer destroys the ImGui context.
    ImGui_ImplGlfw_Shutdown();
}

void ImGuiGLFWRenderer::beginFrame()
{
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

}  // namespace Vulkan
}  // namespace Saiga
#endif
