/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "Renderer.h"

#include "saiga/vulkan/window/Window.h"

namespace Saiga
{
namespace Vulkan
{
VulkanRenderer::VulkanRenderer(VulkanWindow& window, VulkanParameters vulkanParameters)
    : window(window), vulkanParameters(vulkanParameters)
{
    window.setRenderer(this);

    width  = window.getWidth();
    height = window.getHeight();

    std::vector<const char*> instanceExtensions = window.getRequiredInstanceExtensions();
    instance.create(instanceExtensions, vulkanParameters.enableValidationLayer);

    VkSurfaceKHR surface;
    window.createSurface(instance, &surface);

    base.setPhysicalDevice(instance.pickPhysicalDevice());


    vulkanParameters.physicalDeviceFeatures.fillModeNonSolid = VK_TRUE;

    vulkanParameters.physicalDeviceFeatures.wideLines = VK_TRUE;
    base.createLogicalDevice(surface, vulkanParameters.physicalDeviceFeatures, vulkanParameters.deviceExtensions, true,
                             vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer,
                             true);



    base.init(vulkanParameters);

    swapChain.connect(instance, base.physicalDevice, base.device);
    swapChain.initSurface(surface);
    swapChain.create(&width, &height, false);
}

VulkanRenderer::~VulkanRenderer()
{
    // Clean up Vulkan resources
    swapChain.cleanup();

    //    delete base;
    base.destroy();

    instance.destroy();
}

float VulkanRenderer::getTotalRenderTime()
{
    return window.mainLoop.renderCPUTimer.getTimeMS();
}

void VulkanRenderer::initInstanceDevice() {}



}  // namespace Vulkan
}  // namespace Saiga
