#pragma once

#include "saiga/vulkan/svulkan.h"

namespace Saiga {
namespace Vulkan {

class Debug
{
public:
    void init(
            VkInstance instance,
            VkDebugReportFlagsEXT flags = VK_DEBUG_REPORT_WARNING_BIT_EXT  | VK_DEBUG_REPORT_ERROR_BIT_EXT,
            VkDebugReportCallbackEXT callBack = VK_NULL_HANDLE
            );
    void destroy();
    static std::vector<const char*> getDebugValidationLayers();
private:
    vk::Instance instance;

    PFN_vkCreateDebugReportCallbackEXT CreateDebugReportCallback = VK_NULL_HANDLE;
    PFN_vkDestroyDebugReportCallbackEXT DestroyDebugReportCallback = VK_NULL_HANDLE;
    PFN_vkDebugReportMessageEXT dbgBreakCallback = VK_NULL_HANDLE;
    VkDebugReportCallbackEXT msgCallback;
};




}
}
