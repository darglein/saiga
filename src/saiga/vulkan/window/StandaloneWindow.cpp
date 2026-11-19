/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "StandaloneWindow.h"

#include "saiga/vulkan/window/HeadlessWindow.h"
#ifdef SAIGA_USE_GLFW
#    include "saiga/vulkan/window/GLFWWindow.h"
#endif

#include <iostream>
#include <stdexcept>

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

namespace Saiga
{
namespace Vulkan
{
std::unique_ptr<VulkanWindow> CreateVulkanWindow(const WindowParameters& windowParameters)
{
    switch (windowParameters.backend)
    {
        case WindowParameters::Backend::GLFW:
#ifdef SAIGA_USE_GLFW
            return std::make_unique<GLFWWindow>(windowParameters);
#else
            std::cerr << "Saiga was built without GLFW." << std::endl;
            return nullptr;
#endif
        case WindowParameters::Backend::Headless:
            return std::make_unique<HeadlessWindow>(windowParameters);
    }
    return nullptr;
}

static WindowParameters WindowParametersFromConfig(const std::string& file)
{
    WindowParameters params;
    params.fromConfigFile(file);
    return params;
}

static VulkanParameters VulkanParametersFromConfig(const std::string& file)
{
    VulkanParameters params;
    params.fromConfigFile(file);
    return params;
}

StandaloneWindow::StandaloneWindow(const std::string& config_file)
    : StandaloneWindow(WindowParametersFromConfig(config_file), VulkanParametersFromConfig(config_file))
{
    this->config_file = config_file;
}

StandaloneWindow::StandaloneWindow(const WindowParameters& windowParameters, const VulkanParameters& vulkanParameters)
    : vulkan_parameters(vulkanParameters)
{
    window = CreateVulkanWindow(windowParameters);
    if (!window) throw std::runtime_error("Failed to create the Vulkan window");
    window->setUpdateObject(*this);
}

StandaloneWindow::~StandaloneWindow()
{
    // The renderer's Vulkan objects need the window's surface.
    owned_renderer.reset();
    window.reset();
}

void StandaloneWindow::run(const MainLoopParameters& mainLoopParameters)
{
    SAIGA_ASSERT(owned_renderer, "Create the renderer before running the window");
    window->startMainLoop(mainLoopParameters);
    owned_renderer->waitIdle();
}

}  // namespace Vulkan
}  // namespace Saiga
