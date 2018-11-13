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
private:
    vk::BufferUsageFlags usageFlags = vk::BufferUsageFlagBits();
public:
    MemoryLocation m_memoryLocation;

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

    void flush(VulkanBase& base, vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0)
    {
        vk::MappedMemoryRange mappedRange = {};
        mappedRange.memory = m_memoryLocation.memory;
        mappedRange.offset = offset;
        mappedRange.size = size;
        base.device.flushMappedMemoryRanges(mappedRange);
    }

    [[deprecated("This does not allow memory to be cleaned up. Use destroy(VulkanBase&) instead")]]
    void destroy();

    void destroy(VulkanBase& base) {
        if (m_memoryLocation && usageFlags != vk::BufferUsageFlags()) {
            base.memory.getAllocator(usageFlags).deallocate(m_memoryLocation);
        }
    }

//    operator vk::Buffer() const { return m_memoryLocation.buffer; }
};

}
}
