/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/vulkanBase.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL Buffer
{
public:
    size_t size;
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::DescriptorBufferInfo info;

    void createBuffer(
            VulkanBase &base,
            size_t size,
            vk::BufferUsageFlags usage,
            vk::SharingMode sharingMode =  vk::SharingMode::eExclusive
            );

    void allocateMemory(VulkanBase &base);

    uint8_t* map(VulkanBase &base, size_t offset, size_t size);
    void unmap(VulkanBase &base);

    void upload(VulkanBase &base, size_t offset, size_t size, const void* data);
};

}
}
