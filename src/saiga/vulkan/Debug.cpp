/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Debug.h"

#include "saiga/core/util/assert.h"

#include <iostream>

namespace Saiga
{
namespace Vulkan
{
static VKAPI_ATTR VkBool32 VKAPI_CALL messageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                      VkDebugUtilsMessageTypeFlagsEXT type,
                                                      const VkDebugUtilsMessengerCallbackDataEXT* data, void* user_data)
{
    const char* prefix = "INFO";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        prefix = "ERROR";
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        prefix = "WARNING";

    std::cerr << "Vulkan " << prefix;
    if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) std::cerr << " (performance)";
    std::cerr << ": " << (data->pMessageIdName ? data->pMessageIdName : "") << "\n  " << data->pMessage << std::endl;

    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        SAIGA_ASSERT(0, "Vulkan validation error");
    }

    // VK_FALSE: the call that triggered the message is not aborted (VK_TRUE is reserved for layer development).
    return VK_FALSE;
}

vk::DebugUtilsMessengerCreateInfoEXT Debug::createInfo()
{
    vk::DebugUtilsMessengerCreateInfoEXT info;
    info.messageSeverity =
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    info.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    // The parameter type of the callback differs between Vulkan-Hpp versions (C or C++ types), the C function is
    // compatible with both.
    info.pfnUserCallback = reinterpret_cast<decltype(info.pfnUserCallback)>(&messageCallback);
    return info;
}

void Debug::init(vk::Instance _instance)
{
    instance = _instance;

    auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    SAIGA_ASSERT(create, "VK_EXT_debug_utils is not enabled");

    VkDebugUtilsMessengerCreateInfoEXT info = createInfo();
    VkResult err                            = create(instance, &info, nullptr, &messenger);
    SAIGA_ASSERT(err == VK_SUCCESS);
}

void Debug::destroy()
{
    if (messenger != VK_NULL_HANDLE)
    {
        auto destroy_fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_fn) destroy_fn(instance, messenger, nullptr);
        messenger = VK_NULL_HANDLE;
    }
}

std::string Debug::getDebugValidationLayers()
{
    return "VK_LAYER_KHRONOS_validation";
}

}  // namespace Vulkan
}  // namespace Saiga
