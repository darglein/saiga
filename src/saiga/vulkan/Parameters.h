/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"

namespace Saiga
{
namespace Vulkan
{
struct SAIGA_VULKAN_API VulkanParameters
{
    vk::PhysicalDeviceFeatures physicalDeviceFeatures;
    std::vector<const char*> deviceExtensions;
    bool enableValidationLayer = true;
    bool enableImgui           = true;
    bool expand_memory_stats   = false;
    bool enableDefragmentation = false;
    bool enableChunkAllocator  = true;
    uint32_t maxDescriptorSets = 4096 * 4;
    // for {uniformBuffer,texture}
    std::array<uint32_t, 4> descriptorCounts = {4096, 4096, 4096, 4096};



    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);
};


}  // namespace Vulkan
}  // namespace Saiga
