/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#ifndef SAIGA_USE_GLFW
#    error Saiga was compiled without GLFW.
#endif

#include "saiga/core/camera/camera.h"
#include "saiga/core/glfw/glfw_camera.h"
#include "saiga/core/glfw/glfw_eventhandler.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/window/WindowTemplate.h"

namespace Saiga
{
/**
 * The base class for the saiga vulkan samples: a GLFW window with the forward renderer, basic input handling and a
 * controllable camera (WASD + mouse).
 *
 * Keys: ESC closes the window, G toggles the GUI, F12 saves screenshot.png.
 */
class SAIGA_VULKAN_API VulkanExampleBase : public Saiga::Vulkan::StandaloneWindow<Saiga::Vulkan::WindowManagement::GLFW,
                                                                                  Saiga::Vulkan::VulkanForwardRenderer>,
                                           public glfw_KeyListener
{
   public:
    VulkanExampleBase();
    ~VulkanExampleBase() override;

    void update(float dt) override;
    void renderGUI() override;

   protected:
    Glfw_Camera<PerspectiveCamera> camera;

    void keyPressed(int key, int scancode, int mods) override;
};

}  // namespace Saiga
