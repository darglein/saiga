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
    base.bla(physicalDevice);

    VkPhysicalDeviceFeatures enabledFeatures{};
    enabledFeatures.fillModeNonSolid = true;
    enabledFeatures.wideLines = true;
    base.createLogicalDevice(surface,enabledFeatures, enabledDeviceExtensions);


    base.init(vulkanParameters);
    device = base.device;
    cout << endl;



    swapChain.connect(instance, physicalDevice, device);
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

void VulkanRenderer::initInstanceDevice()
{



}



}
}
