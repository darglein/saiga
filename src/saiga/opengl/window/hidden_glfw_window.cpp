/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#if defined(SAIGA_USE_OPENGL) && defined(SAIGA_USE_GLFW)
#    include "saiga/core/util/assert.h"
#    include "saiga/opengl/imgui/imgui_impl_headless_gl3.h"

#    include <GLFW/glfw3.h>
#    include <iostream>

#    include "hidden_glfw_window.h"

namespace Saiga
{
HiddenGLFWWindow::HiddenGLFWWindow(WindowParameters windowParameters, OpenGLParameters openglParameter)
    : OpenGLWindow(windowParameters, openglParameter)
{
    create();
}

HiddenGLFWWindow::~HiddenGLFWWindow()
{
    cleanupSaiga();
    freeContext();
}

std::shared_ptr<ImGuiRenderer> HiddenGLFWWindow::createImGui()
{
    if (windowParameters.imguiParameters.enable)
    {
        return std::make_shared<ImGui_Headless_Renderer>(*this, windowParameters.imguiParameters);
    }
    return {};
}

void HiddenGLFWWindow::setSize(int width, int height)
{
    SAIGA_ASSERT(width > 0 && height > 0);
    if (!window || (width == getWidth() && height == getHeight())) return;
    glfwSetWindowSize(window, width, height);
    int fb_width = 0, fb_height = 0;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);
    if (fb_width != width || fb_height != height)
    {
        std::cerr << "HiddenGLFWWindow: failed to resize to " << width << "x" << height << ", the window is "
                  << fb_width << "x" << fb_height << std::endl;
        if (fb_width <= 0 || fb_height <= 0) return;
    }
    resize(fb_width, fb_height);
}

void HiddenGLFWWindow::checkEvents()
{
    // Keeps the window's message queue empty; a hidden window gets no input.
    glfwPollEvents();
}

void HiddenGLFWWindow::swapBuffers()
{
    glfwSwapBuffers(window);
}

void HiddenGLFWWindow::freeContext()
{
    if (window)
    {
        glfwDestroyWindow(window);
        window = nullptr;
        glfwTerminate();
    }
    // Also called when create() failed before the OpenGL functions were loaded.
    if (OpenGLisInitialized()) terminateOpenGL();
}

bool HiddenGLFWWindow::initWindow()
{
    glfwSetErrorCallback([](int error, const char* description)
                         { std::cerr << "glfw error: " << error << " " << description << std::endl; });
    if (!glfwInit()) return false;

    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, openglParameters.versionMajor);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, openglParameters.versionMinor);
    switch (openglParameters.profile)
    {
        case OpenGLParameters::Profile::ANY:
            break;
        case OpenGLParameters::Profile::CORE:
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            break;
        case OpenGLParameters::Profile::COMPATIBILITY:
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
            break;
    }
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, openglParameters.debug);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, openglParameters.forwardCompatible);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    // The default framebuffer of a visible glfw_Window.
    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 8);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);

    window = glfwCreateWindow(getWidth(), getHeight(), getName().c_str(), nullptr, nullptr);
    if (!window)
    {
        const char* description = nullptr;
        glfwGetError(&description);
        std::cerr << "Failed to create the hidden GLFW window: " << (description ? description : "unknown error")
                  << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(windowParameters.vsync ? 1 : 0);
    glfwGetFramebufferSize(window, &windowParameters.width, &windowParameters.height);
    window_title = getName();
    return true;
}

void HiddenGLFWWindow::loadGLFunctions()
{
    initOpenGL(glfwGetProcAddress);
}

}  // namespace Saiga
#endif
