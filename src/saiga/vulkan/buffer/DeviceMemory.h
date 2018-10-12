/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Base.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL DeviceMemory
{
public:

    ~DeviceMemory() { destroy(); }
    /*
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        bit specifies that memory allocated with this type is the most efficient for device access.
        This property will be set if and only if the memory type belongs to a heap with the VK_MEMORY_HEAP_DEVICE_LOCAL_BIT set.

    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        bit specifies that memory allocated with this type can be mapped for host access using vkMapMemory.

    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        bit specifies that the host cache management commands vkFlushMappedMemoryRanges and
        vkInvalidateMappedMemoryRanges are not needed to flush host writes to the device or make device writes
        visible to the host, respectively.

    VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        bit specifies that memory allocated with this type is cached on the host.
        Host memory accesses to uncached memory are slower than to cached memory, however uncached
        memory is always host coherent.

    VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT
        bit specifies that the memory type only allows device access to the memory.
        Memory types must not have both VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT and VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT set.
        Additionally, the object’s backing memory may be provided by the implementation lazily as specified in Lazily Allocated Memory.

    VK_MEMORY_PROPERTY_PROTECTED_BIT
        bit specifies that the memory type only allows device access to the memory,
        and allows protected queue operations to access the memory. Memory types must not have
        VK_MEMORY_PROPERTY_PROTECTED_BIT set and any of VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT set,
        or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT set, or VK_MEMORY_PROPERTY_HOST_CACHED_BIT set.
        */
    void allocateMemory(
            Saiga::Vulkan::VulkanBase& base,
            const vk::MemoryRequirements& mem_reqs,
            vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );


    /**
     * These map functions only work if "eHostvisible" was set on memory allocation.
     */
    uint8_t* map(size_t offset, size_t size);
    uint8_t* mapAll();
    void unmap();
    void mappedUpload(size_t offset, size_t size, const void* data);
    void mappedDownload(size_t offset, size_t size, void *data);


    size_t getSize() { return size; }
    void destroy();
protected:
    vk::Device device;
    size_t size;
    vk::DeviceMemory memory;

};

}
}
