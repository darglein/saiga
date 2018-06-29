/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/vulkanBase.h"
#include "saiga/vulkan/buffer/DeviceMemory.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL Buffer : public DeviceMemory
{
public:
    vk::Buffer buffer;
    vk::DescriptorBufferInfo info;

    ~Buffer();

    void createBuffer(
            VulkanBase &base,
            size_t size,
            vk::BufferUsageFlags usage,
            vk::SharingMode sharingMode =  vk::SharingMode::eExclusive
            );

    void allocateMemory(VulkanBase &base);
};

}
}
