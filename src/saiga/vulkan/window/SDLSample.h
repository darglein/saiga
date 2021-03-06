/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"

#include "saiga/vulkan/window/WindowTemplate.h"

namespace Saiga
{
/**
 * The base class for the saiga vulkan samples.
 * Includes basic input handling and a controllable camera.
 */
class SAIGA_VULKAN_API VulkanSDLExampleBase
    : public Saiga::Vulkan::StandaloneWindow<Saiga::Vulkan::WindowManagement::SDL,
                                             Saiga::Vulkan::VulkanForwardRenderer>,
      public SDL_KeyListener
{
   public:
    VulkanSDLExampleBase();
    ~VulkanSDLExampleBase() override;

    void update(float dt) override;
    void renderGUI() override;

   protected:
    Glfw_Camera<PerspectiveCamera> camera;


    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};

}  // namespace Saiga
