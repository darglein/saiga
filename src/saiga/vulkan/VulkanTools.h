/*
 * Assorted Vulkan helper functions
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"
#include "saiga/vulkan/VulkanInitializers.hpp"
#include "saiga/vulkan/svulkan.h"

#include <assert.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// Custom define for better code readability
#define VK_FLAGS_NONE 0
// Default fence timeout in nanoseconds
#define DEFAULT_FENCE_TIMEOUT 100000000000

// Macro to check and display Vulkan return results
#if defined(__ANDROID__)
#    define VK_CHECK_RESULT(f)                                                                                        \
        {                                                                                                             \
            VkResult res = (f);                                                                                       \
            if (res != VK_SUCCESS)                                                                                    \
            {                                                                                                         \
                LOGE("Fatal : VkResult is \" %s \" in %s at line %d", vks::tools::errorString(res).c_str(), __FILE__, \
                     __LINE__);                                                                                       \
                assert(res == VK_SUCCESS);                                                                            \
            }                                                                                                         \
        }
#else
#    define VK_CHECK_RESULT(f)                                                                                \
        {                                                                                                     \
            VkResult res = (f);                                                                               \
            if (res != VK_SUCCESS)                                                                            \
            {                                                                                                 \
                std::cout << "Fatal : VkResult is \"" << vks::tools::errorString(res) << "\" in " << __FILE__ \
                          << " at line " << __LINE__ << std::endl;                                            \
                SAIGA_ASSERT(res == VK_SUCCESS);                                                              \
            }                                                                                                 \
        }
#endif

#if defined(__ANDROID__)
#    define ASSET_PATH ""
#else
#    define ASSET_PATH "./../data/"
#endif

namespace vks
{
namespace tools
{
/** @brief Returns an error code as a string */
SAIGA_VULKAN_API std::string errorString(VkResult errorCode);

/** @brief Returns the device type as a string */
SAIGA_VULKAN_API std::string physicalDeviceTypeString(VkPhysicalDeviceType type);

// Selected a suitable supported depth format starting with 32 bit down to 16 bit
// Returns false if none of the depth formats in the list is supported by the device
SAIGA_VULKAN_API VkBool32 getSupportedDepthFormat(VkPhysicalDevice physicalDevice, VkFormat* depthFormat);

}  // namespace tools
}  // namespace vks
