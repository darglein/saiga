/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "Renderer.h"
#include "saiga/vulkan/window/Window.h"

namespace Saiga {
namespace Vulkan {

VulkanRenderer::VulkanRenderer(VulkanWindow &window)
 : window(window)
{
    window.setRenderer(this);

    width = window.getWidth();
    height = window.getHeight();


    initInstanceDevice();


    swapChain.connect(instance, physicalDevice, device);
    VkSurfaceKHR surface;
    window.createSurface(instance,&surface);
    swapChain.initSurface(surface);
    swapChain.create(&width, &height, true);


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

    VkResult err;

    std::vector<const char*> instanceExtensions = window.getRequiredInstanceExtensions();
    //instanceExtensions.push_back( VK_KHR_SURFACE_EXTENSION_NAME );
    instance.create(instanceExtensions,true);



    // Physical device
    uint32_t gpuCount = 0;
    // Get number of available physical devices
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr));
    assert(gpuCount > 0);
    // Enumerate devices
    std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
    err = vkEnumeratePhysicalDevices(instance, &gpuCount, physicalDevices.data());
    if (err) {
        vks::tools::exitFatal("Could not enumerate physical devices : \n" + vks::tools::errorString(err), err);
        return;
    }

    // GPU selection

    // Select physical device to be used for the Vulkan example
    // Defaults to the first device unless specified by command line
    uint32_t selectedDevice = 0;



    physicalDevice = physicalDevices[selectedDevice];

    {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
        std::cout << "Device [" << selectedDevice << "] : " << deviceProperties.deviceName << std::endl;
        std::cout << " Type: " << vks::tools::physicalDeviceTypeString(deviceProperties.deviceType) << std::endl;
        std::cout << " API: " << (deviceProperties.apiVersion >> 22) << "." << ((deviceProperties.apiVersion >> 12) & 0x3ff) << "." << (deviceProperties.apiVersion & 0xfff) << std::endl;
    }


    // Vulkan device creation
    // This is handled by a separate class that gets a logical device representation
    // and encapsulates functions related to a device
    vulkanDevice = new vks::VulkanDevice(physicalDevice);

     VkPhysicalDeviceFeatures enabledFeatures{};
    VkResult res = vulkanDevice->createLogicalDevice(enabledFeatures, enabledDeviceExtensions);
    if (res != VK_SUCCESS) {
        vks::tools::exitFatal("Could not create Vulkan device: \n" + vks::tools::errorString(res), res);
        return;
    }
    device = vulkanDevice->logicalDevice;


}



}
}
