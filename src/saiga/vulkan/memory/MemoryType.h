//
// Created by Peter Eichinger on 08.10.18.
//

#pragma once
#include "saiga/export.h"
#include "MemoryChunk.h"

#include <vulkan/vulkan.hpp>
#include <vector>
namespace Saiga {
namespace Vulkan {
namespace Memory {

/**
 * Abstraction for a single memory type as provided by vkGetPhysicalDeviceMemoryProperties().
 * Allows allocation an deallocation of memory in chunks of the size chunkSize.
 */
class SAIGA_GLOBAL MemoryType {
private:
    uint32_t m_memoryTypeIndex;
    const vk::Device &m_device;
    std::vector<std::shared_ptr<MemoryChunk>> m_chunks;
public:

    vk::MemoryPropertyFlags propertyFlags;

    MemoryType(const MemoryType &memoryType) = delete;

    MemoryType(MemoryType&& memoryType) : m_memoryTypeIndex(memoryType.m_memoryTypeIndex),
        m_device(memoryType.m_device), m_chunks(memoryType.m_chunks),
        propertyFlags(memoryType.propertyFlags){}


    MemoryType(const vk::Device &device, uint32_t memoryTypeIndex, vk::MemoryPropertyFlags flags) :
            m_memoryTypeIndex(memoryTypeIndex), m_device(device), propertyFlags(flags) {
    }

    std::shared_ptr<MemoryChunk> allocate(vk::DeviceSize chunkSize);

    void deallocate(std::shared_ptr<MemoryChunk> chunk);

};


}
}
}


