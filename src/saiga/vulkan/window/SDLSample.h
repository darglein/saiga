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
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/core/window/Interfaces.h"

namespace Saiga
{
/**
 * The base class for the saiga vulkan samples.
 * Includes basic input handling and a controllable camera.
 */
class SAIGA_GLOBAL VulkanSDLExampleBase : public Updating,
                                          public Vulkan::VulkanForwardRenderingInterface,
                                          public SDL_KeyListener
{
   public:
    VulkanSDLExampleBase(Vulkan::VulkanWindow& window, Vulkan::VulkanForwardRenderer& renderer);
    ~VulkanSDLExampleBase() override;

    void update(float dt) override;
    void renderGUI() override;

   protected:
    SDLCamera<PerspectiveCamera> camera;
    Vulkan::VulkanForwardRenderer& renderer;


    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};

}  // namespace Saiga
