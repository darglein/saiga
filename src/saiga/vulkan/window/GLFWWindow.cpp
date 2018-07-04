/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "GLFWWindow.h"
#include <GLFW/glfw3.h>

#if defined(SAIGA_OPENGL_INCLUDED)
#error OpenGL was included somewhere.
#endif


namespace Saiga {
namespace Vulkan {

GLFWWindow::GLFWWindow(WindowParameters _windowParameters)
    :VulkanWindow(_windowParameters)
{
    Saiga::initSaiga(windowParameters.saigaParameters);
    create();
}

std::shared_ptr<ImGuiVulkanRenderer> GLFWWindow::createImGui()
{

//    auto imGui = std::make_shared<Saiga::Vulkan::ImGuiVulkanRenderer>();
    //    imGui->init(sdl_window,(float)windowParameters.width, (float)windowParameters.height);

//    return imGui;
    return nullptr;
}

std::vector<const char *> GLFWWindow::getRequiredInstanceExtensions()
{
    uint32_t count;
    const char** extensions = glfwGetRequiredInstanceExtensions(&count);


    std::vector<const char *> res;
    for(uint32_t i = 0; i < count; ++i)
    {
        res.push_back(extensions[i]);
    }

    return res;
}



void GLFWWindow::create()
{

    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(windowParameters.width, windowParameters.height, "Window Title", NULL, NULL);

    cout << "window created" << endl;


}

void GLFWWindow::createSurface(VkInstance instance, VkSurfaceKHR *surface)
{
    VkResult err = glfwCreateWindowSurface(instance, window, NULL, surface);
    if (err)
    {
        // Window surface creation failed
        SAIGA_ASSERT(0);
    }
}

void GLFWWindow::update(float dt)
{
    glfwPollEvents();
    if(glfwWindowShouldClose(window))
        close();
}



}
}
