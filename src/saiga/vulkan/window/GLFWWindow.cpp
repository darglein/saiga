/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_GLFW

// Declares glfwCreateWindowSurface independent of the include order.
#    define GLFW_INCLUDE_VULKAN
#    include "saiga/core/glfw/glfw_eventhandler.h"
#    include "saiga/core/glfw/saiga_glfw.h"
#    include "saiga/vulkan/imgui/ImGuiGLFWRenderer.h"

#    include "GLFWWindow.h"

#    include <cmath>

#    if defined(SAIGA_OPENGL_INCLUDED)
#        error OpenGL was included somewhere.
#    endif


namespace Saiga
{
namespace Vulkan
{
static void glfwErrorCallback(int error, const char* description)
{
    // GLFW_FEATURE_UNAVAILABLE: e.g. the window position on Wayland (as the OpenGL window does)
    if (error == GLFW_FEATURE_UNAVAILABLE) return;
    std::cout << "GLFW Error " << error << ": " << description << std::endl;
}

GLFWWindow::GLFWWindow(WindowParameters _windowParameters) : VulkanWindow(_windowParameters)
{
    Saiga::initSaiga(windowParameters.saigaParameters);
    create();
}

GLFWWindow::~GLFWWindow()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

std::unique_ptr<ImGuiVulkanRenderer> GLFWWindow::createImGui(size_t frameCount)
{
    if (windowParameters.imguiParameters.enable)
    {
        return std::make_unique<ImGuiGLFWRenderer>(frameCount, windowParameters.imguiParameters, window);
    }
    return nullptr;
}

ivec2 GLFWWindow::getFramebufferSize()
{
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    return ivec2(w, h);
}

void GLFWWindow::setSize(int width, int height)
{
    // GLFW sizes windows in screen coordinates, which differ from pixels with display scaling.
    int window_w, window_h, framebuffer_w, framebuffer_h;
    glfwGetWindowSize(window, &window_w, &window_h);
    glfwGetFramebufferSize(window, &framebuffer_w, &framebuffer_h);
    if (framebuffer_w > 0 && framebuffer_h > 0)
    {
        width  = (int)std::lround(double(width) * window_w / framebuffer_w);
        height = (int)std::lround(double(height) * window_h / framebuffer_h);
    }
    glfwSetWindowSize(window, width, height);
}

std::vector<std::string> GLFWWindow::getRequiredInstanceExtensions()
{
    uint32_t count;
    const char** extensions = glfwGetRequiredInstanceExtensions(&count);


    std::vector<std::string> res;
    for (uint32_t i = 0; i < count; ++i)
    {
        res.push_back(extensions[i]);
    }

    return res;
}



void GLFWWindow::create()
{
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
    {
        SAIGA_ASSERT(0, "glfwInit failed");
        return;
    }

    if (GLFW_FALSE == glfwVulkanSupported())
    {
        std::cout << "Vulkan not supported. Compile GLFW with vulkan!" << std::endl;
        SAIGA_ASSERT(0);
        return;
    }


    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // The swap chain is created once with the window size.
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_VISIBLE, windowParameters.hidden ? GLFW_FALSE : GLFW_TRUE);
    window =
        glfwCreateWindow(windowParameters.width, windowParameters.height, windowParameters.name.c_str(), NULL, NULL);

    SAIGA_ASSERT(window);

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    if (w != windowParameters.width || h != windowParameters.height)
    {
        // E.g. with display scaling.
        std::cout << "GLFW framebuffer size " << w << "x" << h << " differs from the requested window size "
                  << windowParameters.width << "x" << windowParameters.height << std::endl;
    }

    // Input for the keyboard/mouse state and the glfw listeners (e.g. the camera). The ImGui backend chains its own
    // callbacks to these when it is created.
    glfwSetCursorPosCallback(window, glfw_EventHandler::cursor_position_callback);
    glfwSetMouseButtonCallback(window, glfw_EventHandler::mouse_button_callback);
    glfwSetScrollCallback(window, glfw_EventHandler::scroll_callback);
    glfwSetCharCallback(window, glfw_EventHandler::character_callback);
    glfwSetKeyCallback(window, glfw_EventHandler::key_callback);

    std::cout << "GLFW window created." << std::endl;
}

double GLFWWindow::refreshRateHz()
{
    // The monitor that contains the window's center (a full screen window has its own), else the primary one
    GLFWmonitor* monitor = window ? glfwGetWindowMonitor(window) : nullptr;
    // Wayland has no window position (regression: the statistics asked every 0.5 s and GLFW printed an error each
    // time)
    if (!monitor && window && glfwGetPlatform() != GLFW_PLATFORM_WAYLAND)
    {
        int x = 0, y = 0, w = 0, h = 0;
        glfwGetWindowPos(window, &x, &y);
        glfwGetWindowSize(window, &w, &h);
        const int cx = x + w / 2, cy = y + h / 2;
        int count              = 0;
        GLFWmonitor** monitors = glfwGetMonitors(&count);
        for (int i = 0; i < count && !monitor; ++i)
        {
            int mx = 0, my = 0;
            glfwGetMonitorPos(monitors[i], &mx, &my);
            const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
            if (mode && cx >= mx && cx < mx + mode->width && cy >= my && cy < my + mode->height) monitor = monitors[i];
        }
    }
    // A window on Wayland: its monitor is unknown, so is its rate (another monitor's rate would be wrong; the
    // statistics then measure the period from the frames)
    if (!monitor && window && glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) return 0.0;
    if (!monitor) monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = monitor ? glfwGetVideoMode(monitor) : nullptr;
    return mode ? double(mode->refreshRate) : 0.0;
}

void GLFWWindow::createSurface(VkInstance instance, VkSurfaceKHR* surface)
{
    VkResult err = glfwCreateWindowSurface(instance, window, NULL, surface);
    if (err)
    {
        // Window surface creation failed
        std::cout << "Could not create Window Surface!" << std::endl;
        SAIGA_ASSERT(0);
    }
}

void GLFWWindow::update(float dt)
{
    glfwPollEvents();
    // The stored size follows the drawable area, which setSize changes asynchronously on some platforms.
    ivec2 framebuffer = getFramebufferSize();
    if (framebuffer.x() > 0 && framebuffer.y() > 0 && (framebuffer.x() != getWidth() || framebuffer.y() != getHeight()))
    {
        resize(framebuffer.x(), framebuffer.y());
    }
    if (glfwWindowShouldClose(window)) close();
    if (updating) updating->update(dt);
}



}  // namespace Vulkan
}  // namespace Saiga

#endif
