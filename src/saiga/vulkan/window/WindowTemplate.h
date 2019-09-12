/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/window/Interfaces.h"

#include "Window.h"

#ifdef SAIGA_USE_GLFW
#    include "GLFWWindow.h"
#else
namespace Saiga
{
using glfw_Window = void;
}
#endif

#ifdef SAIGA_USE_SDL
#    include "SDLWindow.h"
#else
namespace Saiga
{
using SDLWindow = void;
}
#endif



namespace Saiga
{
namespace Vulkan
{
enum class WindowManagement
{
    SDL,
    GLFW
};

template <WindowManagement WM>
struct GetWindowType
{
};

template <>
struct GetWindowType<WindowManagement::SDL>
{
    using Type = SDLWindow;
};

template <>
struct GetWindowType<WindowManagement::GLFW>
{
    using Type = GLFWWindow;
};


template <WindowManagement WM, typename Renderer>
struct SAIGA_TEMPLATE StandaloneWindow : public Renderer::InterfaceType, public Updating
{
    using WindowManagment = typename GetWindowType<WM>::Type;

    StandaloneWindow(const std::string& config)
    {
        WindowParameters windowParameters;
        Saiga::Vulkan::VulkanParameters vulkanParams;
        typename Renderer::ParameterType rendererParameters;

        windowParameters.fromConfigFile(config);
        vulkanParams.fromConfigFile(config);
        rendererParameters.fromConfigFile(config);


        window   = std::make_unique<WindowManagment>(windowParameters);
        renderer = std::make_unique<Renderer>(*window, vulkanParams);


        window->setUpdateObject(*this);
        renderer->setRenderObject(*this);
    }

    ~StandaloneWindow()
    {
        renderer.reset();
        window.reset();
    }

    void run()
    {
        // Everyhing is initilalized, we can run the main loop now!
        MainLoopParameters mainLoopParameters;
        mainLoopParameters.fromConfigFile("config.ini");
        window->startMainLoop(mainLoopParameters);
        renderer->waitIdle();
    }

    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<WindowManagment> window;
};

}  // namespace Vulkan
}  // namespace Saiga
