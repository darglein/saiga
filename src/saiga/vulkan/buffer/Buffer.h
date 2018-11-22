/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/buffer/DeviceMemory.h"
#include "saiga/vulkan/svulkan.h"

namespace Saiga
{
namespace Vulkan
{
class SAIGA_GLOBAL Buffer
{
   private:
    vk::BufferUsageFlags usageFlags = vk::BufferUsageFlagBits();

   public:
    MemoryLocation m_memoryLocation;

    ~Buffer() { destroy(); }


    void createBuffer(Saiga::Vulkan::VulkanBase& base, size_t size, vk::BufferUsageFlags usage,
                      const vk::MemoryPropertyFlags& flags, vk::SharingMode sharingMode = vk::SharingMode::eExclusive);

    void upload(vk::CommandBuffer& cmd, size_t offset, size_t size, const void* data);

    void stagedUpload(VulkanBase& base, size_t size, const void* data);

    vk::DescriptorBufferInfo createInfo();

    void flush(VulkanBase& base, vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0)
    {
        vk::MappedMemoryRange mappedRange = {};
        mappedRange.memory                = m_memoryLocation.memory;
        mappedRange.offset                = offset;
        mappedRange.size                  = size;
        base.device.flushMappedMemoryRanges(mappedRange);
    }


    [[deprecated("This does not allow memory to be cleaned up. Use destroy(VulkanBase&) instead")]] void destroy();

    void destroy(VulkanBase& base)
    {
        if (m_memoryLocation && usageFlags != vk::BufferUsageFlags())
        {
            base.memory.getAllocator(usageFlags).deallocate(m_memoryLocation);
        }
    }

    vk::DeviceSize size() { return m_memoryLocation.size; }


    void copyTo(vk::CommandBuffer cmd, Buffer& target)
    {
        SAIGA_ASSERT(target.size() >= size());
        vk::BufferCopy bc(m_memoryLocation.offset, target.m_memoryLocation.offset, size());
        cmd.copyBuffer(m_memoryLocation.buffer, target.m_memoryLocation.buffer, bc);
    }

    //    operator vk::Buffer() const { return m_memoryLocation.buffer; }
};

}  // namespace Vulkan
}  // namespace Saiga
