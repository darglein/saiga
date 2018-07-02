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


class VulkanExample : public RenderThing
{
public:
    VulkanExample(Saiga::Vulkan::SDLWindow& window, VulkanForwardRenderer& renderer);
    ~VulkanExample();

    void init();

    virtual void update();
    virtual void render(VkCommandBuffer cmd);
    virtual void renderGUI();
private:
    Saiga::SDLCamera<Saiga::PerspectiveCamera> camera;
    Saiga::Vulkan::Asset teapot;
    Saiga::Vulkan::AssetRenderer assetRenderer;

    VulkanForwardRenderer &renderer;

     bool displayModels = true;
};

