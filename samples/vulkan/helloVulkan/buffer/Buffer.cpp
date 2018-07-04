/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Buffer.h"
#include "vulkanHelper.h"

namespace Saiga {
namespace Vulkan {

Buffer::~Buffer()
{
    device.destroyBuffer(buffer);
}

void Buffer::createBuffer(VulkanBase &base, size_t size, vk::BufferUsageFlags usage, vk::SharingMode sharingMode)
{
    this->size = size;
    cout << "create buffer " << size << endl;
    vk::BufferCreateInfo buf_info = {};
    buf_info.usage = usage;
    buf_info.size = size;
    buf_info.queueFamilyIndexCount = 0;
    buf_info.pQueueFamilyIndices = NULL;
    buf_info.sharingMode = sharingMode;
    CHECK_VK(base.device.createBuffer(&buf_info, NULL, &buffer));

}

void Buffer::allocateMemory(VulkanBase &base)
{
    vk::MemoryRequirements mem_reqs;
    base.device.getBufferMemoryRequirements(buffer, &mem_reqs);

    DeviceMemory::allocateMemory(base,mem_reqs);
}

void Buffer::upload(vk::CommandBuffer &cmd, size_t offset, size_t size, const void *data)
{

    cmd.updateBuffer(buffer,offset,size,data);
}




}
}
