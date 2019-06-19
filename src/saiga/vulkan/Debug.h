#pragma once

#include "saiga/vulkan/svulkan.h"

#include <string>
namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API Debug
{
   public:
    void init(VkInstance instance,
              VkDebugReportFlagsEXT flags       = VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_ERROR_BIT_EXT,
              VkDebugReportCallbackEXT callBack = VK_NULL_HANDLE);
    void destroy();
    static std::string getDebugValidationLayers();

   private:
    vk::Instance instance;

    PFN_vkCreateDebugReportCallbackEXT CreateDebugReportCallback   = VK_NULL_HANDLE;
    PFN_vkDestroyDebugReportCallbackEXT DestroyDebugReportCallback = VK_NULL_HANDLE;
    PFN_vkDebugReportMessageEXT dbgBreakCallback                   = VK_NULL_HANDLE;
    VkDebugReportCallbackEXT msgCallback                           = VK_NULL_HANDLE;
};



}  // namespace Vulkan
}  // namespace Saiga
