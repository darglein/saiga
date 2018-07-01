/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"

namespace Saiga {
namespace Vulkan {

/**
 * This class capsules the platform specific code for window creation.
 */
class SAIGA_GLOBAL VulkanBase
{
public:
    VulkanBase();
    ~VulkanBase();

    void createInstance(bool enableValidation);

    void init_physical_device();
    void createDevice();
public:

    vk::Instance inst;


    // Physical device and some properties of it
    std::vector<vk::PhysicalDevice> physicalDevices;
    vk::PhysicalDevice physicalDevice;
    vk::PhysicalDeviceProperties gpu_props;
    vk::PhysicalDeviceMemoryProperties memory_properties;
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties;


    // Logical device
    vk::Device device;
    vk::Queue queue;

    VkDebugReportCallbackEXT debugReportCallback = nullptr;
};

}
}
