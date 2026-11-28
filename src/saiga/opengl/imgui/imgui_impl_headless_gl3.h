/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui_headless_platform.h"
#include "saiga/opengl/imgui/imgui_opengl.h"

namespace Saiga
{
class OpenGLWindow;

/**
 * ImGui on an OpenGL window without a window system (OffscreenWindow): draws with the OpenGL3 backend into the
 * window's default framebuffer, see ImGuiHeadlessPlatform for input, viewports and the clipboard.
 */
class SAIGA_OPENGL_API ImGui_Headless_Renderer : public ImGuiOpenGLRenderer
{
   public:
    ImGui_Headless_Renderer(OpenGLWindow& window, const ImGuiParameters& params);
    ~ImGui_Headless_Renderer() override;

    void beginFrame() override;
    void render() override;

   private:
    OpenGLWindow& window;
    ImGuiHeadlessPlatform platform;
};

}  // namespace Saiga
