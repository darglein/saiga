/**
 * Copyright (c) 2021 Darius Rückert
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

#ifdef SAIGA_USE_EGL
#    include "saiga/opengl/egl/offscreen_window.h"
#else
namespace Saiga
{
using OffscreenWindow = void;
}
#endif


namespace Saiga
{
enum class WindowManagement
{
    GLFW,
    EGL
};

template <WindowManagement WM>
struct GetWindowType
{
};


template <>
struct GetWindowType<WindowManagement::GLFW>
{
    using Type = glfw_Window;
};

template <>
struct GetWindowType<WindowManagement::EGL>
{
    using Type = OffscreenWindow;
};



template <WindowManagement WM, typename Renderer>
class SAIGA_TEMPLATE StandaloneWindow : public RenderingInterface, public Updating
{
   public:
    using WindowManagment     = typename GetWindowType<WM>::Type;
    using RenderingParameters = typename Renderer::ParameterType;

    StandaloneWindow(const std::string& config)
    {
        config_file = config;

        WindowParameters windowParameters;
        OpenGLParameters openglParameters;
        RenderingParameters rendererParameters;

        windowParameters.fromConfigFile(config);
        openglParameters.fromConfigFile(config);
        rendererParameters.fromConfigFile(config);

        create(windowParameters, openglParameters, rendererParameters);
    }

    StandaloneWindow(std::unique_ptr<Renderer> renderer_, std::unique_ptr<WindowManagment> window_)
        : renderer(std::move(renderer_)), window(std::move(window_))
    {
        window->setUpdateObject(*this);
        renderer->setRenderObject(*this);
    }


    StandaloneWindow(const WindowParameters& windowParameters, const OpenGLParameters& openglParameters,
                     const RenderingParameters& rendererParameters)
    {
        create(windowParameters, openglParameters, rendererParameters);
    }

    ~StandaloneWindow()
    {
        renderer.reset();
        window.reset();
    }

    void run()
    {
        MainLoopParameters mainLoopParameters;
        mainLoopParameters.fromConfigFile(config_file);
        run(mainLoopParameters);
    }

    void run(MainLoopParameters mainLoopParameters) { window->startMainLoop(mainLoopParameters); }

   private:
    void create(const WindowParameters& windowParameters, const OpenGLParameters& openglParameters,
                const RenderingParameters& rendererParameters)
    {
        window   = std::make_unique<WindowManagment>(windowParameters, openglParameters);
        renderer = std::make_unique<Renderer>(*window, rendererParameters);

        window->setUpdateObject(*this);
        renderer->setRenderObject(*this);
    }


   protected:
    std::string config_file;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<WindowManagment> window;
};

}  // namespace Saiga
