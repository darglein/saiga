/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/window/OpenGLWindow.h"

#include <memory>
#include <string>

namespace Saiga
{
/**
 * Creates the window of windowParameters.backend: a glfw_Window or, for Backend::Headless, an OffscreenWindow.
 * Returns nullptr if saiga was built without that backend or the window could not be created (for example without a
 * display or an EGL driver); the reason is printed to std::cerr.
 */
SAIGA_OPENGL_API std::unique_ptr<OpenGLWindow> CreateOpenGLWindow(const WindowParameters& windowParameters,
                                                                  const OpenGLParameters& openglParameters);

/**
 * The base of an OpenGL application: a window, created with the backend chosen at runtime
 * (WindowParameters::backend), and a renderer that renders this object.
 *
 * The derived class creates its renderer in its constructor and keeps the typed pointer it needs:
 *
 *   class App : public StandaloneWindow
 *   {
 *       App() : StandaloneWindow("config.ini") { renderer = CreateRenderer<DeferredRenderer>(); }
 *       DeferredRenderer* renderer;
 *   };
 *
 * The constructors throw std::runtime_error if the window can't be created.
 */
class SAIGA_OPENGL_API StandaloneWindow : public RenderingInterface, public Updating
{
   public:
    // Reads the window and OpenGL parameters from the config file (see CreateRenderer for the renderer's).
    explicit StandaloneWindow(const std::string& config_file);
    StandaloneWindow(const WindowParameters& windowParameters, const OpenGLParameters& openglParameters);
    ~StandaloneWindow() override;

    // The main loop parameters from the config file, or the defaults without one.
    void run();
    void run(MainLoopParameters mainLoopParameters);

   protected:
    // Takes ownership of the renderer and renders this object with it. Once per window: each renderer creates the
    // window's ImGui context.
    template <typename Renderer>
    Renderer* SetRenderer(std::unique_ptr<Renderer> new_renderer)
    {
        SAIGA_ASSERT(!owned_renderer, "The window already has a renderer");
        Renderer* result = new_renderer.get();
        owned_renderer   = std::move(new_renderer);
        owned_renderer->setRenderObject(*this);
        return result;
    }

    template <typename Renderer>
    Renderer* CreateRenderer(const typename Renderer::ParameterType& params)
    {
        return SetRenderer(std::make_unique<Renderer>(*window, params));
    }

    // With the renderer's parameters from the config file this window was created with, if any.
    template <typename Renderer>
    Renderer* CreateRenderer()
    {
        typename Renderer::ParameterType params;
        if (!config_file.empty()) params.fromConfigFile(config_file);
        return CreateRenderer<Renderer>(params);
    }

    std::string config_file;
    std::unique_ptr<OpenGLWindow> window;

   private:
    std::unique_ptr<OpenGLRenderer> owned_renderer;
};

}  // namespace Saiga
