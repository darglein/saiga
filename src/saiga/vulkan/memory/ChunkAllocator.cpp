//
// Created by Peter Eichinger on 08.10.18.
//

#include "ChunkAllocator.h"
using Saiga::Vulkan::Memory::MemoryLocation;
using Saiga::Vulkan::Memory::ChunkAllocator;
using Saiga::Vulkan::Memory::ChunkType;
using Saiga::Vulkan::Memory::Chunk;

ChunkType& ChunkAllocator::findMemoryType(vk::MemoryPropertyFlags flags) {
    for(auto& memType : m_memoryTypes) {
        if ((memType.propertyFlags & flags) == flags) {
            return memType;
        }
    }
    throw std::runtime_error("No valid memory found");
}


void Saiga::Vulkan::Memory::ChunkAllocator::init(vk::PhysicalDevice _physicalDevice, vk::Device _device) {
    m_physicalDevice = _physicalDevice;
    m_device = _device;

    auto memoryProperties = m_physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        auto& properties = memoryProperties.memoryTypes[i];

        m_memoryTypes.push_back(std::move(ChunkType(m_device,i,properties.propertyFlags)));
    }
    m_initialized = true;
}

std::shared_ptr<Chunk> ChunkAllocator::allocate(vk::MemoryPropertyFlags propertyFlags,vk::DeviceSize chunkSize) {
    if (!m_initialized) {
        throw std::runtime_error("Must be initialized before use");
    }
    auto& memType = findMemoryType(propertyFlags);

    return memType.allocate(chunkSize);
}

void Saiga::Vulkan::Memory::ChunkAllocator::deallocate(std::shared_ptr<Chunk> chunk) {
    if (!m_initialized) {
        throw std::runtime_error("Must be initialized before use");
    }
    findMemoryType(chunk->flags).deallocate(chunk);
}
