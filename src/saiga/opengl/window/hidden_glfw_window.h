/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/opengl/window/OpenGLWindow.h"

struct GLFWwindow;

namespace Saiga
{
/**
 * The headless OpenGL window where there is no EGL (Windows): a GLFW window that is never shown, whose default
 * framebuffer is the window's (read it with ScreenshotDefaultFramebuffer()). WGL needs a window for a context, so
 * unlike OffscreenWindow it needs a desktop session, but nothing appears on screen or in the taskbar.
 *
 * As in OffscreenWindow, ImGui runs without a platform backend (ImGui_Headless_Renderer) and the window takes no input
 * from the window system: the application adds it to ImGui's input queue. Without vsync, limit the frame rate with
 * MainLoopParameters::framesPerSecond. Usually created through WindowParameters::Backend::Headless.
 */
class SAIGA_OPENGL_API HiddenGLFWWindow : public OpenGLWindow
{
   public:
    HiddenGLFWWindow(WindowParameters windowParameters, OpenGLParameters openglParameter = OpenGLParameters());
    ~HiddenGLFWWindow() override;

    std::shared_ptr<ImGuiRenderer> createImGui() override;
    void setSize(int width, int height) override;

   protected:
    bool initInput() override { return true; }
    void checkEvents() override;
    void swapBuffers() override;
    void freeContext() override;
    bool initWindow() override;
    void loadGLFunctions() override;

   private:
    GLFWwindow* window = nullptr;
};

}  // namespace Saiga
