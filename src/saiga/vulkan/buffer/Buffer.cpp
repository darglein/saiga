/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Buffer.h"

namespace Saiga {
namespace Vulkan {

void Buffer::destroy()
{
    DeviceMemory::destroy();
    if(buffer)
        device.destroyBuffer(buffer);
}

void Buffer::createBuffer(vks::VulkanDevice *vulkanDevice, size_t size, vk::BufferUsageFlags usage, vk::SharingMode sharingMode)
{
    this->device = vulkanDevice->logicalDevice;
    this->size = size;
    vk::BufferCreateInfo buf_info = {};
    buf_info.usage = usage;
    buf_info.size = size;
    buf_info.queueFamilyIndexCount = 0;
    buf_info.pQueueFamilyIndices = NULL;
    buf_info.sharingMode = sharingMode;
    CHECK_VK(device.createBuffer(&buf_info, NULL, &buffer));
}

void Buffer::allocateMemoryBuffer(vks::VulkanDevice *vulkanDevice, vk::MemoryPropertyFlags flags)
{
    SAIGA_ASSERT(buffer);
    vk::MemoryRequirements mem_reqs;
    device.getBufferMemoryRequirements(buffer, &mem_reqs);
    DeviceMemory::allocateMemory(vulkanDevice,mem_reqs,flags);
    device.bindBufferMemory(buffer,memory,0);
}

void Buffer::upload(vk::CommandBuffer &cmd, size_t offset, size_t size, const void *data)
{
    cmd.updateBuffer(buffer,offset,size,data);
}

vk::DescriptorBufferInfo Buffer::createInfo()
{
    vk::DescriptorBufferInfo info(
                buffer,0,size
                );
    return info;
}




}
}
