/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifdef SAIGA_USE_GLFW

#    include "GLFWWindow.h"

#    include "saiga/core/glfw/saiga_glfw.h"

#    if defined(SAIGA_OPENGL_INCLUDED)
#        error OpenGL was included somewhere.
#    endif


namespace Saiga
{
namespace Vulkan
{
static void printGLFWerror()
{
    //    const char* description;
    //    glfwGetError(&description);
    //    std::cout << "GLFW Error: " << description << std::endl;
    SAIGA_ASSERT(0);
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
    // TODO: Create GLFW imgui
    return nullptr;
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
    if (!glfwInit())
    {
        printGLFWerror();
        return;
    }

    if (GLFW_FALSE == glfwVulkanSupported())
    {
        std::cout << "Vulkan not supported. Compile GLFW with vulkan!" << std::endl;
        SAIGA_ASSERT(0);
        return;
    }


    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window =
        glfwCreateWindow(windowParameters.width, windowParameters.height, windowParameters.name.c_str(), NULL, NULL);

    SAIGA_ASSERT(window);

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    SAIGA_ASSERT(w == windowParameters.width && h == windowParameters.height);

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
