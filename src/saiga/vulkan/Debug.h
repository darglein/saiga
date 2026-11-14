/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/svulkan.h"

#include <string>

namespace Saiga
{
namespace Vulkan
{
/**
 * Validation layer output through VK_EXT_debug_utils.
 * Warnings are printed, errors are printed and fail an assertion (with SAIGA_ASSERTS on).
 */
class SAIGA_VULKAN_API Debug
{
   public:
    ~Debug() { destroy(); }

    // The create info for the messenger. Also chained into the instance create info, so that messages from
    // vkCreateInstance/vkDestroyInstance are reported as well.
    static vk::DebugUtilsMessengerCreateInfoEXT createInfo();

    void init(vk::Instance instance);
    void destroy();

    static std::string getDebugValidationLayers();

   private:
    vk::Instance instance;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
};

}  // namespace Vulkan
}  // namespace Saiga
