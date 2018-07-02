/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Instance.h"

#include "base/VulkanDebug.h"

namespace Saiga {
namespace Vulkan {

void Instance::create(std::vector<const char*> instanceExtensions, bool enableValidation)
{


    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "test";
    appInfo.pEngineName = "test engine";
    appInfo.apiVersion = VK_API_VERSION_1_1;

//    std::vector<const char*> instanceExtensions = getRequiredInstanceExtensions();

//    instanceExtensions.push_back( VK_KHR_SURFACE_EXTENSION_NAME );




    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pNext = NULL;
    instanceCreateInfo.pApplicationInfo = &appInfo;
    if (instanceExtensions.size() > 0)
    {
        if (enableValidation)
        {
            instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }
        instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
        instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
    }
    if (enableValidation)
    {
        instanceCreateInfo.enabledLayerCount = vks::debug::validationLayerCount;
        instanceCreateInfo.ppEnabledLayerNames = vks::debug::validationLayerNames;
    }



    vkCreateInstance(&instanceCreateInfo, nullptr, &instance);



    // If requested, we enable the default validation layers for debugging
    if (enableValidation)
    {
        // The report flags determine what type of messages for the layers will be displayed
        // For validating (debugging) an appplication the error and warning bits should suffice
        VkDebugReportFlagsEXT debugReportFlags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
        // Additional flags include performance info, loader and layer debug messages, etc.
        vks::debug::setupDebugging(instance, debugReportFlags, VK_NULL_HANDLE);
    }

}

}
}
