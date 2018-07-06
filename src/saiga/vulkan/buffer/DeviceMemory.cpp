/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "DeviceMemory.h"


namespace Saiga {
namespace Vulkan {


void DeviceMemory::destroy()
{
//    SAIGA_ASSERT(device);
//    SAIGA_ASSERT(memory);
    if(device && memory)
        device.freeMemory(memory);
}

void DeviceMemory::allocateMemory(VulkanBase &base, const vk::MemoryRequirements &mem_reqs, vk::MemoryPropertyFlags flags)
{
    device = base;
    SAIGA_ASSERT(device);
    //    vk::MemoryRequirements mem_reqs;
    //    base.device.getBufferMemoryRequirements(buffer, &mem_reqs);

    vk::MemoryAllocateInfo alloc_info = {};
    //        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    //        alloc_info.pNext = NULL;
    alloc_info.memoryTypeIndex = 0;

    alloc_info.allocationSize = mem_reqs.size;
//    bool pass = Vulkan::memory_type_from_properties(
//                vulkanDevicememory_properties,mem_reqs.memoryTypeBits,
//                flags,
//                &alloc_info.memoryTypeIndex);
    alloc_info.memoryTypeIndex  = base.getMemoryType(mem_reqs.memoryTypeBits,(VkMemoryPropertyFlags)flags);

//    SAIGA_ASSERT(pass);

    CHECK_VK(device.allocateMemory(&alloc_info,nullptr,&memory));
}

uint8_t *DeviceMemory::map(size_t offset, size_t size)
{
    uint8_t *pData;
    device.mapMemory(memory, 0, size, vk::MemoryMapFlags(), (void **)&pData);
    return pData;
}

void DeviceMemory::unmap()
{
    device.unmapMemory(memory);
}

void DeviceMemory::mappedUpload(size_t offset, size_t size, const void *data)
{
    uint8_t *pData = map(offset,size) ;
    memcpy(pData, data, size);
    unmap();
}

void DeviceMemory::mappedDownload(size_t offset, size_t size, void *data)
{
    uint8_t *pData = map(offset,size) ;
    memcpy(data,pData, size);
    unmap();
}

}
}
