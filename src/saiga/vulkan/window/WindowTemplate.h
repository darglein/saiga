/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/window/Interfaces.h"

#include "HeadlessWindow.h"
#include "Window.h"

#ifdef SAIGA_USE_GLFW
#    include "GLFWWindow.h"
#endif



namespace Saiga
{
namespace Vulkan
{
enum class WindowManagement
{
    GLFW,
    // No window system, see HeadlessWindow.
    Headless
};

template <WindowManagement WM>
struct GetWindowType
{
};

#ifdef SAIGA_USE_GLFW
template <>
struct GetWindowType<WindowManagement::GLFW>
{
    using Type = GLFWWindow;
};
#endif


template <>
struct GetWindowType<WindowManagement::Headless>
{
    using Type = HeadlessWindow;
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

        init(windowParameters, vulkanParams, rendererParameters);
    }

    StandaloneWindow(const WindowParameters& windowParameters, const VulkanParameters& vulkanParams,
                     const typename Renderer::ParameterType& rendererParameters)
    {
        init(windowParameters, vulkanParams, rendererParameters);
    }

    ~StandaloneWindow()
    {
        renderer.reset();
        window.reset();
    }

    void run(const MainLoopParameters& mainLoopParameters = MainLoopParameters())
    {
        // Everyhing is initilalized, we can run the main loop now!
        window->startMainLoop(mainLoopParameters);
        renderer->waitIdle();
    }

    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<WindowManagment> window;

   private:
    void init(const WindowParameters& windowParameters, const VulkanParameters& vulkanParams,
              const typename Renderer::ParameterType& rendererParameters)
    {
        window   = std::make_unique<WindowManagment>(windowParameters);
        renderer = std::make_unique<Renderer>(*window, vulkanParams);

        window->setUpdateObject(*this);
        renderer->setRenderObject(*this);
    }
};

}  // namespace Vulkan
}  // namespace Saiga
