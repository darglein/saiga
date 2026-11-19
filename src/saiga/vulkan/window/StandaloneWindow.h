/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/core/window/MainLoop.h"
#include "saiga/vulkan/Parameters.h"
#include "saiga/vulkan/Renderer.h"
#include "saiga/vulkan/window/Window.h"

#include <memory>
#include <string>

namespace Saiga
{
namespace Vulkan
{
/**
 * Creates the window of windowParameters.backend: a GLFWWindow or, for Backend::Headless, a HeadlessWindow.
 * Returns nullptr if saiga was built without that backend; the reason is printed to std::cerr. A GLFWWindow that
 * can't be created (no display) fails an assertion.
 */
SAIGA_VULKAN_API std::unique_ptr<VulkanWindow> CreateVulkanWindow(const WindowParameters& windowParameters);

/**
 * The base of a Vulkan application: a window, created with the backend chosen at runtime
 * (WindowParameters::backend), and a renderer.
 *
 * The derived class also derives from the rendering interface of its renderer, creates the renderer in its
 * constructor and keeps the typed pointer it needs:
 *
 *   class App : public StandaloneWindow, public VulkanForwardRenderingInterface
 *   {
 *       App() : StandaloneWindow("config.ini") { renderer = CreateRenderer<VulkanForwardRenderer>(*this); }
 *       VulkanForwardRenderer* renderer;
 *   };
 *
 * The constructors throw std::runtime_error if saiga was built without the backend.
 */
class SAIGA_VULKAN_API StandaloneWindow : public Updating
{
   public:
    // Reads the window and Vulkan parameters from the config file (see CreateRenderer for the renderer's).
    explicit StandaloneWindow(const std::string& config_file);
    StandaloneWindow(const WindowParameters& windowParameters, const VulkanParameters& vulkanParameters);
    ~StandaloneWindow() override;

    // Runs until the window is closed and waits for the GPU afterwards.
    void run(const MainLoopParameters& mainLoopParameters = MainLoopParameters());

   protected:
    // Takes ownership of the renderer, which renders the given interface (usually this object). Once per window: each
    // renderer creates the window's ImGui context.
    template <typename Renderer>
    Renderer* SetRenderer(std::unique_ptr<Renderer> new_renderer, RenderingInterface& rendering)
    {
        SAIGA_ASSERT(!owned_renderer, "The window already has a renderer");
        Renderer* result = new_renderer.get();
        owned_renderer   = std::move(new_renderer);
        owned_renderer->setRenderObject(rendering);
        return result;
    }

    template <typename Renderer>
    Renderer* CreateRenderer(RenderingInterface& rendering, const typename Renderer::ParameterType& params)
    {
        return SetRenderer(std::make_unique<Renderer>(*window, vulkan_parameters, params), rendering);
    }

    // With the renderer's parameters from the config file this window was created with, if any.
    template <typename Renderer>
    Renderer* CreateRenderer(RenderingInterface& rendering)
    {
        typename Renderer::ParameterType params;
        if (!config_file.empty()) params.fromConfigFile(config_file);
        return CreateRenderer<Renderer>(rendering, params);
    }

    std::string config_file;
    std::unique_ptr<VulkanWindow> window;

   private:
    VulkanParameters vulkan_parameters;
    std::unique_ptr<VulkanRenderer> owned_renderer;
};

}  // namespace Vulkan
}  // namespace Saiga
