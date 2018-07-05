/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "Renderer.h"
#include "saiga/vulkan/window/Window.h"

namespace Saiga {
namespace Vulkan {

VulkanRenderer::VulkanRenderer(VulkanWindow &window, VulkanParameters vulkanParameters)
    : window(window), vulkanParameters(vulkanParameters)
{
    window.setRenderer(this);

    width = window.getWidth();
    height = window.getHeight();

    std::vector<const char*> instanceExtensions = window.getRequiredInstanceExtensions();
    instance.create(instanceExtensions,vulkanParameters.enableValidationLayer);

    VkSurfaceKHR surface;
    window.createSurface(instance,&surface);

    physicalDevice = instance.pickPhysicalDevice();


    // Vulkan device creation
    // This is handled by a separate class that gets a logical device representation
    // and encapsulates functions related to a device
    vulkanDevice = new vks::VulkanDevice(physicalDevice);

    VkPhysicalDeviceFeatures enabledFeatures{};
    enabledFeatures.fillModeNonSolid = true;
    enabledFeatures.wideLines = true;
    VkResult res = vulkanDevice->createLogicalDevice(surface,enabledFeatures, enabledDeviceExtensions);
    if (res != VK_SUCCESS) {
        vks::tools::exitFatal("Could not create Vulkan device: \n" + vks::tools::errorString(res), res);
        return;
    }
    device = vulkanDevice->logicalDevice;
    cout << endl;



    swapChain.connect(instance, physicalDevice, device);
    swapChain.initSurface(surface);
    swapChain.create(&width, &height, false);


    VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
}

VulkanRenderer::~VulkanRenderer()
{
    // Clean up Vulkan resources
    swapChain.cleanup();

    delete vulkanDevice;

    instance.destroy();



}

void VulkanRenderer::initInstanceDevice()
{



}



}
}
