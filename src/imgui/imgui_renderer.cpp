/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/imgui/imgui_renderer.h"
#include "saiga/imgui/imgui.h"

namespace Saiga {

void ImGuiRenderer::checkWindowFocus()
{
//    isFocused |= ImGui::IsWindowFocused();
    wantsCaptureMouse |= ImGui::GetIO().WantCaptureMouse;
}

void ImGuiRenderer::endFrame()
{
     ImGui::Render();
}


}
