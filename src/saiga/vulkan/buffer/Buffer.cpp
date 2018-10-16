/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Buffer.h"
#include "saiga/util/imath.h"
#include "saiga/vulkan/buffer/StagingBuffer.h"

namespace Saiga {
namespace Vulkan {

void Buffer::destroy()
{

}

void Buffer::createBuffer(Saiga::Vulkan::VulkanBase& base, size_t size, vk::BufferUsageFlags usage, vk::SharingMode sharingMode)
{
    m_memoryLocation = base.memory.getAllocator(usage).allocate(size);
}

void Buffer::upload(vk::CommandBuffer &cmd, size_t offset, size_t size, const void *data)
{
    size = iAlignUp(size,4);
    cmd.updateBuffer(m_memoryLocation.buffer,m_memoryLocation.offset,size,data);
}

/**
 * Perform a staged upload to the buffer. A StagingBuffer is created and used for this.
 * \see Saiga::Vulkan::StagingBuffer
 * @param base A reference to a VulkanBase
 * @param size Size of the data
 * @param data Pointer to the data.
 */
void Buffer::stagedUpload(VulkanBase &base, size_t size, const void *data)
{
    vk::CommandBuffer cmd = base.createAndBeginTransferCommand();

    StagingBuffer staging;
    staging.init(base,size,data);

    vk::BufferCopy bc(0,m_memoryLocation.offset,size);
    cmd.copyBuffer(staging.m_memoryLocation.buffer,m_memoryLocation.buffer,bc);

    base.endTransferWait(cmd);
}

vk::DescriptorBufferInfo Buffer::createInfo()
{
    return {m_memoryLocation.buffer,m_memoryLocation.offset, m_memoryLocation.size};
}

}
}
