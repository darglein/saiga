/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ImGuiHeadlessRenderer.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/window/WindowBase.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

namespace Saiga
{
namespace Vulkan
{
ImGuiHeadlessRenderer::ImGuiHeadlessRenderer(size_t frameCount, const ImGuiParameters& params, WindowBase& window)
    : ImGuiVulkanRenderer(frameCount, params), window(window)
{
}

void ImGuiHeadlessRenderer::beginFrame()
{
    ImGui::SetCurrentContext(imguiContext());
    platform.NewFrame(window.getWidth(), window.getHeight());
    ImGui::NewFrame();
}

}  // namespace Vulkan
}  // namespace Saiga
