/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Buffer.h"

#include "saiga/core/math/imath.h"
#include "saiga/vulkan/buffer/StagingBuffer.h"

#include <cstring>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
void Buffer::destroy()
{
    if (!m_memoryLocation) return;

    // The device is gone after VulkanBase::destroy() (the allocator was destroyed with it).
    if (base && base->device)
    {
        // Frames in flight may still use the buffer.
        auto b        = base;
        auto type     = Memory::BufferType{bufferUsage, memoryProperties};
        auto location = m_memoryLocation;
        base->deferDelete([b, type, location]() { b->memory.deallocateBuffer(type, location); });
    }
    m_memoryLocation = nullptr;
}

void Buffer::destroyImmediately()
{
    if (!m_memoryLocation) return;
    if (base && base->device)
    {
        base->memory.deallocateBuffer({bufferUsage, memoryProperties}, m_memoryLocation);
    }
    m_memoryLocation = nullptr;
}

void Buffer::createBuffer(Saiga::Vulkan::VulkanBase& base, size_t size, vk::BufferUsageFlags bufferUsage,
                          const vk::MemoryPropertyFlags& memoryProperties, vk::SharingMode sharingMode)
{
    // TODO: Sharing mode is not used yet
    destroy();
    this->base             = &base;
    this->bufferUsage      = bufferUsage;
    this->memoryProperties = memoryProperties;
    m_memoryLocation       = base.memory.allocate({this->bufferUsage, this->memoryProperties}, size);
    SAIGA_ASSERT(m_memoryLocation, "Buffer allocation failed");
}

void Buffer::barrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                     vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess) const
{
    SAIGA_ASSERT(m_memoryLocation);
    vk::BufferMemoryBarrier2 b;
    b.srcStageMask        = srcStage;
    b.srcAccessMask       = srcAccess;
    b.dstStageMask        = dstStage;
    b.dstAccessMask       = dstAccess;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    {
        Memory::SafeAccessor acc(*m_memoryLocation);
        b.buffer = m_memoryLocation->data;
        b.offset = m_memoryLocation->offset;
        b.size   = m_memoryLocation->size;
    }

    vk::DependencyInfo dep;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers    = &b;
    cmd.pipelineBarrier2(dep);
}

void Buffer::stagedUpload(VulkanBase& base, size_t size, const void* data)
{
    SAIGA_ASSERT(m_memoryLocation && size <= this->size(), "stagedUpload: the data is larger than the buffer");
    if (size == 0) return;

    // The host write to the (coherent) staging buffer is made visible by the queue submission.
    StagingBuffer staging;
    staging.init(base, size, data);

    vk::CommandBuffer cmd = base.mainQueue.commandPool.createAndBeginOneTimeBuffer();

    // Previous commands on the queue may still read/write the buffer.
    barrier(cmd, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    staging.copyTo(cmd, *this, 0, 0, size);
    // Make the upload visible to all later commands.
    barrier(cmd, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);

    cmd.end();
    base.mainQueue.submitAndWait(cmd);
    base.mainQueue.commandPool.freeCommandBuffer(cmd);

    staging.destroyImmediately();
}

void Buffer::stagedDownload(void* data)
{
    SAIGA_ASSERT(base && m_memoryLocation);

    StagingBuffer staging;
    staging.init(*base, size());

    vk::CommandBuffer cmd = base->mainQueue.commandPool.createAndBeginOneTimeBuffer();

    // Wait for previous writes to the buffer.
    barrier(cmd, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    copyTo(cmd, staging);
    // The host reads the staging buffer after the fence wait.
    staging.barrier(cmd, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);

    cmd.end();
    base->mainQueue.submitAndWait(cmd);
    base->mainQueue.commandPool.freeCommandBuffer(cmd);

    staging.download(data);
    staging.destroyImmediately();
}

vk::DescriptorBufferInfo Buffer::createInfo()
{
    return {m_memoryLocation->data, m_memoryLocation->offset, m_memoryLocation->size};
}

void Buffer::flush(VulkanBase& base, vk::DeviceSize size, vk::DeviceSize offset)
{
    vk::MappedMemoryRange mappedRange = {};
    // The buffer is sub-allocated from a larger memory block, so the range is relative to the location.
    mappedRange.memory = m_memoryLocation->memory;
    mappedRange.offset = m_memoryLocation->offset + offset;
    mappedRange.size   = size == VK_WHOLE_SIZE ? m_memoryLocation->size - offset : size;

    // Offset and size must be multiples of nonCoherentAtomSize (or the size must reach the end of the memory).
    auto atom          = base.physicalDevice.getProperties().limits.nonCoherentAtomSize;
    auto end           = mappedRange.offset + mappedRange.size;
    mappedRange.offset = (mappedRange.offset / atom) * atom;
    mappedRange.size   = ((end - mappedRange.offset + atom - 1) / atom) * atom;
    // Rounding up stays inside the memory block for sub-allocations (chunk sizes are multiples of the atom size). A
    // dedicated allocation (offset 0) can end before the rounded end: flush to its end.
    if (m_memoryLocation->offset == 0 && mappedRange.offset + mappedRange.size > m_memoryLocation->size)
    {
        mappedRange.size = VK_WHOLE_SIZE;
    }
    base.device.flushMappedMemoryRanges(mappedRange);
}

void Buffer::download(void* data, size_t size)
{
    SAIGA_ASSERT(size <= this->size());
    std::vector<char> tmp(this->size());
    download(tmp.data());
    std::memcpy(data, tmp.data(), size);
}

void Buffer::copyTo(vk::CommandBuffer cmd, Buffer& target, vk::DeviceSize srcOffset, vk::DeviceSize dstOffset,
                    vk::DeviceSize size)
{
    SAIGA_ASSERT(m_memoryLocation && target.m_memoryLocation);
    SAIGA_ASSERT(srcOffset <= this->size() && dstOffset <= target.size());
    if (size == VK_WHOLE_SIZE)
    {
        size = this->size() - srcOffset;
    }
    SAIGA_ASSERT(this->size() - srcOffset >= size, "Source buffer is not large enough");
    SAIGA_ASSERT(target.size() - dstOffset >= size, "Destination buffer is not large enough");
    Memory::SafeAccessor acc(*m_memoryLocation, *target.m_memoryLocation);
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

void Buffer::update(vk::CommandBuffer cmd, size_t size, const void* data, vk::DeviceSize offset)
{
    // Limits of vkCmdUpdateBuffer
    SAIGA_ASSERT(size <= 65536 && size % 4 == 0 && offset % 4 == 0, "Use a staged upload for this buffer update");
    SAIGA_ASSERT(m_memoryLocation && offset + size <= this->size(), "Buffer::update: out of bounds");
    Memory::SafeAccessor acc(*m_memoryLocation);
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
