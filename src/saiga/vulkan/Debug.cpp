/*
 * Vulkan examples debug wrapper
 *
 * Appendix for VK_EXT_Debug_Report can be found at
 * https://github.com/KhronosGroup/Vulkan-Docs/blob/1.0-VK_EXT_debug_report/doc/specs/vulkan/appendices/debug_report.txt
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "Debug.h"

#include <iostream>
namespace Saiga
{
namespace Vulkan
{
static const std::vector<std::string> typeNames = {
    "VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT                     ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_INSTANCE_EXT                    ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT             ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT                      ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT                       ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_SEMAPHORE_EXT                   ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT              ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_FENCE_EXT                       ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT               ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT                      ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT                       ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_EVENT_EXT                       ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_QUERY_POOL_EXT                  ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_VIEW_EXT                 ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_VIEW_EXT                  ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT               ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_CACHE_EXT              ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT             ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT                 ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT                    ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT       ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT                     ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT             ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT              ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT                 ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_POOL_EXT                ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT                 ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_SWAPCHAIN_KHR_EXT               ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT_EXT   ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_KHR_EXT                 ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_MODE_KHR_EXT            ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_OBJECT_TABLE_NVX_EXT            ",
    "VK_DEBUG_REPORT_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NVX_EXT",
    "VK_DEBUG_REPORT_OBJECT_TYPE_VALIDATION_CACHE_EXT_EXT        ",
};



VkBool32 messageCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t srcObject,
                         size_t location, int32_t msgCode, const char* pLayerPrefix, const char* pMsg, void* pUserData)
{
    // Select prefix depending on flags passed to the callback
    // Note that multiple flags may be set for a single validation message
    std::string prefix("");

    // Error that may result in undefined behaviour
    if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
    {
        prefix += "ERROR";
    };
    // Warnings may hint at unexpected / non-spec API usage
    if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
    {
        prefix += "WARNING";
    };
    // May indicate sub-optimal usage of the API
    if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)
    {
        prefix += "PERFORMANCE";
    };
    // Informal messages that may become handy during debugging
    if (flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT)
    {
        prefix += "INFO";
    }
    // Diagnostic info from the Vulkan loader and layers
    // Usually not helpful in terms of API usage, but may help to debug layer and loader problems
    if (flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT)
    {
        prefix += "DEBUG";
    }

    std::string typestring = (int)objType < typeNames.size() ? typeNames[objType] : std::to_string((int)objType);

    // Display message to default output (console/logcat)

    std::cerr << "Vulkan " << pLayerPrefix << " callback" << std::endl
              << "  Severity    : " << prefix << std::endl
              << "  Code        : " << msgCode << std::endl
              << "  Object ID   : " << srcObject << std::endl
              << "  Object Type : " << typestring << std::endl
              << "  Location    : " << location << std::endl
              << "  Message     : " << pMsg << std::endl
              << std::endl;


    SAIGA_ASSERT(0);
    // The return value of this callback controls wether the Vulkan call that caused
    // the validation message will be aborted or not
    // We return VK_FALSE as we DON'T want Vulkan calls that cause a validation message
    // (and return a VkResult) to abort
    // If you instead want to have calls abort, pass in VK_TRUE and the function will
    // return VK_ERROR_VALIDATION_FAILED_EXT
    return VK_TRUE;
}

void Debug::init(VkInstance _instance, VkDebugReportFlagsEXT flags, VkDebugReportCallbackEXT callBack)
{
    instance = _instance;

    CreateDebugReportCallback = reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT"));
    DestroyDebugReportCallback = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
    dbgBreakCallback =
        reinterpret_cast<PFN_vkDebugReportMessageEXT>(vkGetInstanceProcAddr(instance, "vkDebugReportMessageEXT"));

    VkDebugReportCallbackCreateInfoEXT dbgCreateInfo = {};
    dbgCreateInfo.sType                              = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
    dbgCreateInfo.pfnCallback                        = (PFN_vkDebugReportCallbackEXT)messageCallback;
    dbgCreateInfo.flags                              = flags;

    VkResult err = CreateDebugReportCallback(instance, &dbgCreateInfo, nullptr,
                                             (callBack != VK_NULL_HANDLE) ? &callBack : &msgCallback);
    SAIGA_ASSERT(!err);
}

void Debug::destroy()
{
    if (msgCallback != VK_NULL_HANDLE)
    {
        DestroyDebugReportCallback(instance, msgCallback, nullptr);
    }
}

std::string Debug::getDebugValidationLayers()
{
    //    return "VK_LAYER_LUNARG_standard_validation";
    return "VK_LAYER_KHRONOS_validation";
}


}  // namespace Vulkan
}  // namespace Saiga
