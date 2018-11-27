/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Buffer.h"
#include "saiga/util/imath.h"
#include "saiga/vulkan/buffer/StagingBuffer.h"

namespace Saiga
{
namespace Vulkan
{
void Buffer::destroy()
{
    if (m_memoryLocation && usageFlags != vk::BufferUsageFlags())
    {
        base->memory.getAllocator(usageFlags).deallocate(m_memoryLocation);
        m_memoryLocation.buffer = nullptr;
    }
}

void Buffer::createBuffer(Saiga::Vulkan::VulkanBase& base, size_t size, vk::BufferUsageFlags usage,
                          const vk::MemoryPropertyFlags& flags, vk::SharingMode sharingMode)
{
    // TODO: Sharing mode is not used yet
    this->base = &base;
    m_memoryLocation = base.memory.getAllocator(usage, flags).allocate(size);
    usageFlags       = usage;
    if (size != m_memoryLocation.size)
    {
        LOG(WARNING) << "Unequal sizes " << size << " " << m_memoryLocation.size;
    }
}


void Buffer::stagedUpload(VulkanBase& base, size_t size, const void* data)
{
    vk::CommandBuffer cmd = base.createAndBeginTransferCommand();

    StagingBuffer staging;
    staging.init(base, size, data);

    vk::BufferCopy bc(0, m_memoryLocation.offset, size);
    cmd.copyBuffer(staging.m_memoryLocation.buffer, m_memoryLocation.buffer, bc);

    base.endTransferWait(cmd);
}

vk::DescriptorBufferInfo Buffer::createInfo()
{
    return {m_memoryLocation.buffer, m_memoryLocation.offset, m_memoryLocation.size};
}

void Buffer::flush(VulkanBase &base, vk::DeviceSize size, vk::DeviceSize offset) {
    vk::MappedMemoryRange mappedRange = {};
    mappedRange.memory                = m_memoryLocation.memory;
    mappedRange.offset                = offset;
    mappedRange.size                  = size;
    base.device.flushMappedMemoryRanges(mappedRange);
}

void Buffer::copyTo(vk::CommandBuffer cmd, Buffer &target, vk::DeviceSize srcOffset, vk::DeviceSize dstOffset,
                    vk::DeviceSize size) {
    if (size == VK_WHOLE_SIZE)
    {
        size = this->size() - srcOffset;
    }
    SAIGA_ASSERT(this->size() - srcOffset >= size, "Source buffer is not large enough");
    SAIGA_ASSERT(target.size() - dstOffset >= size, "Destination buffer is not large enough");
    vk::BufferCopy bc{m_memoryLocation.offset + srcOffset, target.m_memoryLocation.offset + dstOffset, size};
    cmd.copyBuffer(m_memoryLocation.buffer, target.m_memoryLocation.buffer, bc);
}

void Buffer::copyTo(vk::CommandBuffer cmd, vk::Image dstImage, vk::ImageLayout dstImageLayout,
                    vk::ArrayProxy<const vk::BufferImageCopy> regions) {
    cmd.copyBufferToImage(m_memoryLocation.buffer, dstImage, dstImageLayout, regions);
}

vk::BufferImageCopy Buffer::getBufferImageCopy(vk::DeviceSize offset) const { return {m_memoryLocation.offset + offset}; }

void Buffer::update(vk::CommandBuffer cmd, size_t size, void *data, vk::DeviceSize offset) {
    cmd.updateBuffer(m_memoryLocation.buffer, m_memoryLocation.offset + offset, size, data);
}

}  // namespace Vulkan
}  // namespace Saiga
