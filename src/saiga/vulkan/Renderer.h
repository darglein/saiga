/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"
#include "saiga/window/Interfaces.h"

namespace Saiga {
namespace Vulkan {

class VulkanWindow;

class SAIGA_GLOBAL VulkanRenderer : public RendererBase
{
public:

    VulkanRenderer(VulkanWindow &window);
    virtual ~VulkanRenderer();

    virtual void render(Camera *cam) {}
    virtual void bindCamera(Camera* cam){}
};


class SAIGA_GLOBAL VulkanForwardRenderingInterface : public RenderingBase
{
public:
    VulkanForwardRenderingInterface(RendererBase& parent) : RenderingBase(parent) {}
    virtual ~VulkanForwardRenderingInterface(){}

    virtual void render(VkCommandBuffer cmd) {}
    virtual void renderGUI() {}
};


}
}
