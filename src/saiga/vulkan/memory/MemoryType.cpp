//
// Created by Peter Eichinger on 08.10.18.
//

#include "MemoryType.h"

using Saiga::Vulkan::Memory::MemoryType;
using Saiga::Vulkan::Memory::MemoryChunk;
std::shared_ptr<MemoryChunk> MemoryType::allocate(vk::DeviceSize chunkSize) {
    vk::MemoryAllocateInfo info(chunkSize, m_memoryTypeIndex);
    auto chunk = std::make_shared<MemoryChunk>(m_device.allocateMemory(info), chunkSize, propertyFlags);
    m_chunks.push_back(chunk);
    return chunk;
}

void MemoryType::deallocate(std::shared_ptr<MemoryChunk> chunk) {
    m_device.free(chunk->memory);
    m_chunks.erase(std::remove(m_chunks.begin(), m_chunks.end(), chunk), m_chunks.end());
}
