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

#    if defined(SAIGA_OPENGL_INCLUDED)
#        error OpenGL was included somewhere.
#    endif


namespace Saiga
{
namespace Vulkan
{
static void glfwErrorCallback(int error, const char* description)
{
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
    if (glfwWindowShouldClose(window)) close();
    if (updating) updating->update(dt);
}



}  // namespace Vulkan
}  // namespace Saiga

#endif
