/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"
#include "saiga/window/Interfaces.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/Device.h"
#include "saiga/vulkan/ImGuiVulkanRenderer.h"
#include "saiga/vulkan/VulkanSwapChain.hpp"

namespace Saiga {
namespace Vulkan {

class VulkanWindow;

class SAIGA_GLOBAL VulkanRenderer : public RendererBase
{
public:

    VkDevice device;
    vks::VulkanDevice *vulkanDevice;
    VkPipelineCache pipelineCache;


    VulkanRenderer(VulkanWindow &window);
    virtual ~VulkanRenderer();

    virtual void render(Camera *cam) {}
    virtual void bindCamera(Camera* cam){}
protected:

    /**
     * Shared Member variables common for all vulkan render engines.
     */

    Saiga::Vulkan::VulkanWindow& window;


    uint32_t width = 1280;
    uint32_t height = 720;

    Saiga::Vulkan::Instance instance;
    VkPhysicalDevice physicalDevice;


    std::vector<const char*> enabledDeviceExtensions;
    std::vector<const char*> enabledInstanceExtensions;
    VulkanSwapChain swapChain;

private:
    void initInstanceDevice();
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
