/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/window/Interfaces.h"

#include "OpenGLWindow.h"

#ifdef SAIGA_USE_GLFW
#    include "glfw_window.h"
#else
namespace Saiga
{
using glfw_Window = void;
}
#endif

#ifdef SAIGA_USE_SDL
#    include "sdl_window.h"
#else
namespace Saiga
{
using SDLWindow = void;
}
#endif



namespace Saiga
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
    using Type = glfw_Window;
};


template <WindowManagement WM, typename Renderer>
class SAIGA_TEMPLATE StandaloneWindow : public Renderer::InterfaceType, public Updating
{
   public:
    using WindowManagment     = typename GetWindowType<WM>::Type;
    using RenderingParameters = typename Renderer::ParameterType;

    StandaloneWindow(const std::string& config)
    {
        WindowParameters windowParameters;
        OpenGLParameters openglParameters;
        RenderingParameters rendererParameters;

        windowParameters.fromConfigFile(config);
        openglParameters.fromConfigFile(config);
        rendererParameters.fromConfigFile(config);
        mainLoopParameters.fromConfigFile(config);

        create(windowParameters, openglParameters, rendererParameters);
    }

    ~StandaloneWindow()
    {
        renderer.reset();
        window.reset();
    }

    void run() { window->startMainLoop(mainLoopParameters); }

   private:
    void create(const WindowParameters& windowParameters, const OpenGLParameters& openglParameters,
                const RenderingParameters& rendererParameters)
    {
        window   = std::make_unique<WindowManagment>(windowParameters, openglParameters);
        renderer = std::make_unique<Renderer>(*window, rendererParameters);


        window->setUpdateObject(*this);
        renderer->setRenderObject(*this);
    }

    MainLoopParameters mainLoopParameters;

   protected:
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<WindowManagment> window;
};

}  // namespace Saiga
