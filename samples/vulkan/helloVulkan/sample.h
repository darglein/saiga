/*
* Vulkan Example - imGui (https://github.com/ocornut/imgui)
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/AssetRenderer.h"
#include "saiga/sdl/sdl_camera.h"
#include "saiga/window/Interfaces.h"


class VulkanExample :  public Saiga::Updating, public Saiga::Vulkan::VulkanForwardRenderingInterface, public Saiga::SDL_KeyListener
{
public:
    VulkanExample(
            Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer
            );
    ~VulkanExample();

    virtual void init(Saiga::Vulkan::Queue& queue, vk::CommandBuffer cmd);

    virtual void update(float dt) override;
        virtual void transfer(VkCommandBuffer cmd);
    virtual void render(VkCommandBuffer cmd) override;
    virtual void renderGUI() override;
private:
    Saiga::SDLCamera<Saiga::PerspectiveCamera> camera;

    Saiga::Object3D teapotTrans;
    Saiga::Vulkan::VulkanVertexColoredAsset teapot,plane;
    Saiga::Vulkan::AssetRenderer assetRenderer;

    Saiga::Vulkan::VulkanForwardRenderer &renderer;

    bool displayModels = true;


    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};

