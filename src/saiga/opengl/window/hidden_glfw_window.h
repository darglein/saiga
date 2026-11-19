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
// The headless window where there is no EGL (Windows): a GLFW window that is never shown and gets no input. WGL needs
// a window for its context, so unlike the EGL window it needs a logged-in desktop session.
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
