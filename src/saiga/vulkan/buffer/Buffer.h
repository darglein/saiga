/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/buffer/DeviceMemory.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL Buffer
{
public:
    MemoryLocation m_memoryLocation;
public:

    ~Buffer() { destroy(); }


    void createBuffer(
            Saiga::Vulkan::VulkanBase& base,
            size_t size,
            vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eUniformBuffer,
            vk::SharingMode sharingMode =  vk::SharingMode::eExclusive
            );

    void upload(
            vk::CommandBuffer& cmd,
            size_t offset,
            size_t size,
            const void* data
            );

    void stagedUpload(VulkanBase &base, size_t size, const void *data);

    vk::DescriptorBufferInfo createInfo();

    void destroy();

//    operator vk::Buffer() const { return m_memoryLocation.buffer; }
};

}
}
