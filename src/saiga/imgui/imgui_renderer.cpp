/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/imgui/imgui_renderer.h"

#include "saiga/imgui/imgui.h"

namespace Saiga
{
ImGuiRenderer::ImGuiRenderer(const ImGuiParameters& params)
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
