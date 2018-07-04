/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "UniformBuffer.h"
#include "vulkanHelper.h"

namespace Saiga {
namespace Vulkan {

void UniformBuffer::init(VulkanBase &base, size_t size)
{
    createBuffer(base,size,vk::BufferUsageFlagBits::eUniformBuffer|vk::BufferUsageFlagBits::eTransferDst);
    allocateMemory(base);
    info.buffer = buffer;
    info.offset = 0;
    info.range = size;
    base.device.bindBufferMemory(buffer,memory,0);
}




}
}
