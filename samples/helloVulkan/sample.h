/*
* Vulkan Example - imGui (https://github.com/ocornut/imgui)
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "saiga/vulkan/base/vulkanexamplebase.h"
#include "saiga/vulkan/AssetRenderer.h"
#include "saiga/sdl/sdl_camera.h"
#include "saiga/window/Interfaces.h"

class VulkanExample :  public Saiga::Updating, public Saiga::Vulkan::VulkanForwardRenderingInterface
{
public:
    VulkanExample(Saiga::Vulkan::SDLWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer);
    ~VulkanExample();

    void init();

    virtual void update(float dt) override;
    virtual void render(VkCommandBuffer cmd) override;
    virtual void renderGUI() override;
private:
    Saiga::SDLCamera<Saiga::PerspectiveCamera> camera;
    Saiga::Vulkan::Asset teapot;
    Saiga::Vulkan::AssetRenderer assetRenderer;

    Saiga::Vulkan::VulkanForwardRenderer &renderer;

     bool displayModels = true;
};

