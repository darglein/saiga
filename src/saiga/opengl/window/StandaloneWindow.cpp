/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "StandaloneWindow.h"

#include "saiga/core/window/MainLoop.h"
#ifdef SAIGA_USE_GLFW
#    include "saiga/opengl/window/glfw_window.h"
#endif
#ifdef SAIGA_USE_EGL
#    include "saiga/opengl/egl/offscreen_window.h"
#endif

#include <iostream>
#include <stdexcept>

namespace Saiga
{
std::unique_ptr<OpenGLWindow> CreateOpenGLWindow(const WindowParameters& windowParameters,
                                                 const OpenGLParameters& openglParameters)
{
    std::unique_ptr<OpenGLWindow> window;
    try
    {
        switch (windowParameters.backend)
        {
            case WindowParameters::Backend::GLFW:
#ifdef SAIGA_USE_GLFW
                window = std::make_unique<glfw_Window>(windowParameters, openglParameters);
#else
                std::cerr << "Saiga was built without GLFW." << std::endl;
#endif
                break;
            case WindowParameters::Backend::Headless:
#ifdef SAIGA_USE_EGL
                window = std::make_unique<OffscreenWindow>(windowParameters, openglParameters);
#else
                std::cerr << "Saiga was built without EGL, which headless OpenGL windows need." << std::endl;
#endif
                break;
        }
    }
    catch (const std::exception& e)
    {
        // glfw_Window throws if GLFW can't create the window.
        std::cerr << "Failed to create the window: " << e.what() << std::endl;
        return nullptr;
    }
    if (window && !window->isCreated()) window.reset();
    return window;
}

static WindowParameters WindowParametersFromConfig(const std::string& file)
{
    WindowParameters params;
    params.fromConfigFile(file);
    return params;
}

static OpenGLParameters OpenGLParametersFromConfig(const std::string& file)
{
    OpenGLParameters params;
    params.fromConfigFile(file);
    return params;
}

StandaloneWindow::StandaloneWindow(const std::string& config_file)
    : StandaloneWindow(WindowParametersFromConfig(config_file), OpenGLParametersFromConfig(config_file))
{
    this->config_file = config_file;
}

StandaloneWindow::StandaloneWindow(const WindowParameters& windowParameters, const OpenGLParameters& openglParameters)
{
    window = CreateOpenGLWindow(windowParameters, openglParameters);
    if (!window)
    {
        throw std::runtime_error(windowParameters.backend == WindowParameters::Backend::Headless
                                     ? "Failed to create the headless OpenGL window"
                                     : "Failed to create the OpenGL window");
    }
    window->setUpdateObject(*this);
}

StandaloneWindow::~StandaloneWindow()
{
    // The renderer's OpenGL objects need the window's context.
    owned_renderer.reset();
    window.reset();
}

void StandaloneWindow::run()
{
    MainLoopParameters mainLoopParameters;
    if (!config_file.empty()) mainLoopParameters.fromConfigFile(config_file);
    run(mainLoopParameters);
}

void StandaloneWindow::run(MainLoopParameters mainLoopParameters)
{
    SAIGA_ASSERT(owned_renderer, "Create the renderer before running the window");
    window->startMainLoop(mainLoopParameters);
}

}  // namespace Saiga
