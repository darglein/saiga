/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Buffer.h"

#include "saiga/core/math/imath.h"
#include "saiga/vulkan/buffer/StagingBuffer.h"

namespace Saiga
{
namespace Vulkan
{
void Buffer::destroy()
{
    if (m_memoryLocation && bufferUsage != vk::BufferUsageFlags())
    {
        base->memory.deallocateBuffer({bufferUsage, memoryProperties}, m_memoryLocation);
        m_memoryLocation = nullptr;
    }
}

void Buffer::createBuffer(Saiga::Vulkan::VulkanBase& base, size_t size, vk::BufferUsageFlags bufferUsage,
                          const vk::MemoryPropertyFlags& memoryProperties, vk::SharingMode sharingMode)
{
    // TODO: Sharing mode is not used yet
    this->base             = &base;
    this->bufferUsage      = bufferUsage;
    this->memoryProperties = memoryProperties;
    m_memoryLocation       = base.memory.allocate({this->bufferUsage, this->memoryProperties}, size);
}


void Buffer::stagedUpload(VulkanBase& base, size_t size, const void* data)
{
    vk::CommandBuffer cmd = base.mainQueue.commandPool.createAndBeginOneTimeBuffer();

    StagingBuffer staging;
    staging.init(base, size, data);

    copy_buffer(cmd, m_memoryLocation, staging.m_memoryLocation);

    cmd.end();
    base.mainQueue.submitAndWait(cmd);
}

void Buffer::stagedDownload(void* data)
{
    vk::CommandBuffer cmd = base->mainQueue.commandPool.createAndBeginOneTimeBuffer();

    StagingBuffer staging;
    staging.init(*base, size());

    copy_buffer(cmd, staging.m_memoryLocation, m_memoryLocation);

    cmd.end();
    base->mainQueue.submitAndWait(cmd);

    staging.download(data);
}

vk::DescriptorBufferInfo Buffer::createInfo()
{
    return {m_memoryLocation->data, m_memoryLocation->offset, m_memoryLocation->size};
}

void Buffer::flush(VulkanBase& base, vk::DeviceSize size, vk::DeviceSize offset)
{
    vk::MappedMemoryRange mappedRange = {};
    mappedRange.memory                = m_memoryLocation->memory;
    mappedRange.offset                = offset;
    mappedRange.size                  = size;
    base.device.flushMappedMemoryRanges(mappedRange);
}

void Buffer::copyTo(vk::CommandBuffer cmd, Buffer& target, vk::DeviceSize srcOffset, vk::DeviceSize dstOffset,
                    vk::DeviceSize size)
{
    if (size == VK_WHOLE_SIZE)
    {
        size = this->size() - srcOffset;
    }
    SAIGA_ASSERT(this->size() - srcOffset >= size, "Source buffer is not large enough");
    SAIGA_ASSERT(target.size() - dstOffset >= size, "Destination buffer is not large enough");
    vk::BufferCopy bc{m_memoryLocation->offset + srcOffset, target.m_memoryLocation->offset + dstOffset, size};
    cmd.copyBuffer(m_memoryLocation->data, target.m_memoryLocation->data, bc);
}

void Buffer::copyTo(vk::CommandBuffer cmd, vk::Image dstImage, vk::ImageLayout dstImageLayout,
                    vk::ArrayProxy<const vk::BufferImageCopy> regions)
{
    Memory::SafeAccessor acc(*m_memoryLocation);
    cmd.copyBufferToImage(m_memoryLocation->data, dstImage, dstImageLayout, regions);
}

vk::BufferImageCopy Buffer::getBufferImageCopy(vk::DeviceSize offset) const
{
    Memory::SafeAccessor acc(*m_memoryLocation);
    return {m_memoryLocation->offset + offset};
}

void Buffer::update(vk::CommandBuffer cmd, size_t size, void* data, vk::DeviceSize offset)
{
    cmd.updateBuffer(m_memoryLocation->data, m_memoryLocation->offset + offset, size, data);
}

std::ostream& operator<<(std::ostream& os, const Buffer& buffer)
{
    os << " bufferUsage: " << vk::to_string(buffer.bufferUsage)
       << " memoryProperties: " << vk::to_string(buffer.memoryProperties) << " location: " << buffer.m_memoryLocation;
    return os;
}

}  // namespace Vulkan
}  // namespace Saiga
