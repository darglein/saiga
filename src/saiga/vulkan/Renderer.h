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



    VulkanRenderer(VulkanWindow &window);
    virtual ~VulkanRenderer();

    virtual void render(Camera *cam) {}
    virtual void bindCamera(Camera* cam){}
public:

    /**
     * Shared Member variables common for all vulkan render engines.
     */

    Saiga::Vulkan::VulkanWindow& window;


    uint32_t width = 1280;
    uint32_t height = 720;


    /** @brief Encapsulated physical and logical vulkan device */
    vks::VulkanDevice *vulkanDevice;

    Saiga::Vulkan::Instance instance;
    // Physical device (GPU) that Vulkan will ise
    VkPhysicalDevice physicalDevice;


    std::vector<const char*> enabledDeviceExtensions;
    std::vector<const char*> enabledInstanceExtensions;
    /** @brief Logical device, application's view of the physical device (GPU) */
    // todo: getter? should always point to VulkanDevice->device
    VkDevice device;

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
