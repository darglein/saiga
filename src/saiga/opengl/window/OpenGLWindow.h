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

    // False if create() failed, for example without a display or without a usable OpenGL driver.
    bool isCreated() const { return created; }

    void renderImGui(bool* p_open = nullptr);


    // reading the default framebuffer
    TemplatedImage<ucvec4> ScreenshotDefaultFramebuffer();



    virtual std::shared_ptr<ImGuiRenderer> createImGui() { return nullptr; }

    // Operations of a window of the window system. A window without one (headless) keeps the title, ignores the
    // icon, attention, raising and maximizing, and is always shown.
    virtual void setWindowTitle(const std::string& title) { window_title = title; }
    const std::string& getWindowTitle() const { return window_title; }
    virtual void setWindowIcon(Image* image) {}
    virtual void requestAttention() {}
    virtual void bringToFront() {}
    virtual void maximize() {}
    virtual float getContentScale() { return windowParameters.headlessContentScale; }
    // False while the window is hidden or minimized: its default framebuffer then holds no pixels of its own.
    virtual bool isShown() { return true; }
    // Resizes the drawable area to the given size in pixels.
    virtual void setSize(int width, int height) = 0;

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
    bool created             = false;
    std::string window_title;
};

}  // namespace Saiga
