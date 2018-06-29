/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Buffer.h"
#include "saiga/vulkan/vulkanHelper.h"

namespace Saiga {
namespace Vulkan {

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

    vk::MemoryAllocateInfo alloc_info = {};
//        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
//        alloc_info.pNext = NULL;
    alloc_info.memoryTypeIndex = 0;

    alloc_info.allocationSize = mem_reqs.size;
    bool pass = Vulkan::memory_type_from_properties(base.memory_properties,mem_reqs.memoryTypeBits,
                                       vk::MemoryPropertyFlagBits::eHostVisible| vk::MemoryPropertyFlagBits::eHostCoherent,
                                       &alloc_info.memoryTypeIndex);

    SAIGA_ASSERT(pass);

    CHECK_VK(base.device.allocateMemory(&alloc_info,nullptr,&memory));
}

uint8_t *Buffer::map(VulkanBase &base, size_t offset, size_t size)
{
    uint8_t *pData;
    base.device.mapMemory(memory, 0, size, vk::MemoryMapFlags(), (void **)&pData);
    return pData;
}

void Buffer::unmap(VulkanBase &base)
{
    base.device.unmapMemory(memory);
}

void Buffer::upload(VulkanBase &base, size_t offset, size_t size, const void *data)
{
    uint8_t *pData = map(base,offset,size) ;

    memcpy(pData, data, size);

    unmap(base);
}



}
}
