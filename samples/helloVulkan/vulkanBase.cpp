/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vulkanBase.h"

namespace Saiga {
namespace Vulkan {

VulkanBase::VulkanBase()
{

}

VulkanBase::~VulkanBase()
{

}


// Debug callback for validation layer messages
VkBool32 debugMessageCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t srcObject, size_t location, int32_t msgCode, const char* pLayerPrefix, const char* pMsg, void* pUserData) {
    // Select prefix depending on flags passed to the callback
    // Note that multiple flags may be set for a single validation message
    std::string prefix("");

    // Error that may result in undefined behaviour
    if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
    {
        prefix += "ERROR:";
    };
    // Warnings may hint at unexpected / non-spec API usage
    if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
    {
        prefix += "WARNING:";
    };
    // May indicate sub-optimal usage of the API
    if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)
    {
        prefix += "PERFORMANCE:";
    };
    // Informal messages that may become handy during debugging
    if (flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT)
    {
        prefix += "INFO:";
    }
    // Diagnostic info from the Vulkan loader and layers
    // Usually not helpful in terms of API usage, but may help to debug layer and loader problems
    if (flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT)
    {
        prefix += "DEBUG:";
    }

    std::cout << prefix << " [" << pLayerPrefix << "] Code " << msgCode << " : " << pMsg << std::endl;

    SAIGA_ASSERT(0);

    return VK_FALSE;
}

void VulkanBase::createInstance(bool enableValidation)
{
    vk::ApplicationInfo appInfo;
    appInfo.pApplicationName = "Vulkan Tutorial 01";
    appInfo.pEngineName = "VK_TUTORIAL";
    appInfo.apiVersion = VK_API_VERSION_1_0;

    std::vector<const char*> enabledExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };

    // Enable surface extensions depending on os
#if defined(_WIN32)
    enabledExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(__ANDROID__)
    enabledExtensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(_DIRECT2DISPLAY)
    enabledExtensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
#elif defined(__linux__)
    enabledExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif

    if (enableValidation) {
        enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    vk::InstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.pApplicationInfo = &appInfo;
    instanceCreateInfo.enabledExtensionCount = (uint32_t)enabledExtensions.size();
    instanceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
    if (enableValidation) {
        instanceCreateInfo.enabledLayerCount = 1;
        const char *validationLayerNames[] = { "VK_LAYER_LUNARG_standard_validation" };
        instanceCreateInfo.ppEnabledLayerNames = validationLayerNames;
    }

    inst = vk::createInstance(instanceCreateInfo);

    // Setup debug callback to display validation layer messages
    if (enableValidation) {
        auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(inst, "vkCreateDebugReportCallbackEXT");

        if (vkCreateDebugReportCallbackEXT) {
            vk::DebugReportCallbackCreateInfoEXT dbgCreateInfo;
            dbgCreateInfo.pfnCallback = (PFN_vkDebugReportCallbackEXT)debugMessageCallback;
            dbgCreateInfo.flags = vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning;
            VkResult res = vkCreateDebugReportCallbackEXT(inst, reinterpret_cast<const VkDebugReportCallbackCreateInfoEXT*>(&dbgCreateInfo), nullptr, &debugReportCallback);
            assert(res == VK_SUCCESS);
        }

    }
}


void VulkanBase::init_physical_device()
{
    // ======================= Physical Devices =======================


    {
        // Print all physical devices and choose first one.
        physicalDevices = inst.enumeratePhysicalDevices();
        SAIGA_ASSERT(physicalDevices.size() >= 1);
        for(vk::PhysicalDevice& d : physicalDevices)
        {
            vk::PhysicalDeviceProperties props = d.getProperties();
            cout << "[Device] Id=" << props.deviceID << " "  << props.deviceName << " Type=" << (int)props.deviceType << endl;
        }
        physicalDevice = physicalDevices[0];
    }

    {

        cout << "Creating a device from physical id " << physicalDevice.getProperties().deviceID << "." << endl;
        queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        SAIGA_ASSERT(queueFamilyProperties.size() >= 1);
        for(vk::QueueFamilyProperties& qf : queueFamilyProperties)
        {
            cout << "[QueueFamily] Count=" << qf.queueCount << " flags=" << (unsigned int)qf.queueFlags << endl;
        }
    }

    /* This is as good a place as any to do this */
    //    vkGetPhysicalDeviceMemoryProperties(info.gpus[0], &info.memory_properties);
    //    vkGetPhysicalDeviceProperties(info.gpus[0], &info.gpu_props);
    memory_properties = physicalDevice.getMemoryProperties();
    gpu_props = physicalDevice.getProperties();

}

void VulkanBase::createDevice()
{
    // Request one graphics queue

    // Get index of first queue family that supports graphics


    uint32_t graphicsQueueFamilyIndex = 0;
    for (size_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics)
        {
            graphicsQueueFamilyIndex = i;
            break;
        }
    }

    const float defaultQueuePriority(0.0f);

    vk::DeviceQueueCreateInfo queueCreatInfo;
    queueCreatInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
    queueCreatInfo.queueCount = 1;
    queueCreatInfo.pQueuePriorities = &defaultQueuePriority;

    // Create the logical device representation
    std::vector<const char*> deviceExtensions;
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    vk::DeviceCreateInfo deviceCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreatInfo;
    // No specific features used in this tutorial
    deviceCreateInfo.pEnabledFeatures = nullptr;

    if (deviceExtensions.size() > 0)
    {
        deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    }

    device = physicalDevice.createDevice(deviceCreateInfo);

    // Get a graphics queue from the device
    queue = device.getQueue(graphicsQueueFamilyIndex, 0);
}


}
}
