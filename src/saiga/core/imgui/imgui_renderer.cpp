/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui_renderer.h"

#include "saiga/core/imgui/imgui.h"

namespace Saiga
{
ImGuiRenderer::ImGuiRenderer(ImGuiParameters params)
{
    ImGui::CreateContext();
    Saiga::initImGui(params);
}

ImGuiRenderer::~ImGuiRenderer()
{
    ImGui::DestroyContext();
}



void ImGuiRenderer::endFrame()
{
    ImGui::Render();
}

// void ImGuiRenderer::render()
//{
//    renderDrawLists(ImGui::GetDrawData());
//}


}  // namespace Saiga
