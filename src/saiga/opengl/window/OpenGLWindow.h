/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/window/WindowBase.h"
#include "saiga/opengl/imgui/imgui_opengl.h"
#include "saiga/opengl/opengl_helper.h"

namespace Saiga
{
class Camera;
class Image;

class SAIGA_OPENGL_API OpenGLWindow : public WindowBase
{
   public:
    OpenGLWindow(WindowParameters windowParameters, OpenGLParameters openglParameters);
    virtual ~OpenGLWindow();

    bool create();
    void destroy();

    void renderImGui(bool* p_open = nullptr);


    // reading the default framebuffer
    TemplatedImage<ucvec4> ScreenshotDefaultFramebuffer();



    virtual std::shared_ptr<ImGui_GL_Renderer> createImGui() { return nullptr; }

    virtual void swap();
    virtual void update(float dt);

    OpenGLParameters openglParameters;

   protected:
    virtual bool initWindow() = 0;
    virtual bool initInput()  = 0;
    virtual bool shouldClose() { return !running; }
    virtual void checkEvents()     = 0;
    virtual void swapBuffers()     = 0;
    virtual void freeContext()     = 0;
    virtual void loadGLFunctions() = 0;

    void sleep(tick_t ticks);

    bool auto_reload_shaders = false;
};

}  // namespace Saiga
