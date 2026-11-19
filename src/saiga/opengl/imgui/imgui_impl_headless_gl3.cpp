/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imgui_impl_headless_gl3.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_main_menu.h"
#include "saiga/opengl/window/OpenGLWindow.h"

#include "internal/imgui_impl_opengl3.h"

namespace Saiga
{
ImGui_Headless_Renderer::ImGui_Headless_Renderer(OpenGLWindow& window, const ImGuiParameters& params)
    : ImGuiRenderer(params), window(window)
{
    ImGui_ImplOpenGL3_Init();
}

ImGui_Headless_Renderer::~ImGui_Headless_Renderer()
{
    ImGui::SetCurrentContext(imguiContext());
    ImGui_ImplOpenGL3_Shutdown();
}

void ImGui_Headless_Renderer::beginFrame()
{
    ImGui::SetCurrentContext(imguiContext());
    // Creates the device objects, the font texture among them, on the first frame.
    ImGui_ImplOpenGL3_NewFrame();
    ImGuiIO& io = ImGui::GetIO();
    if (!io.Fonts->IsBuilt())
    {
        // The font settings changed: replace the texture of the old atlas.
        ImGui_ImplOpenGL3_DestroyFontsTexture();
        ImGui_ImplOpenGL3_CreateFontsTexture();
    }

    int w = window.getWidth();
    int h = window.getHeight();
    platform.NewFrame(w, h);
    ImGui::NewFrame();

    editor_gui.render(w, h);
}

void ImGui_Headless_Renderer::render()
{
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

}  // namespace Saiga
