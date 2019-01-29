//
// Created by Peter Eichinger on 08.10.18.
//

#include "ChunkType.h"

#include "SafeAllocator.h"
namespace Saiga::Vulkan::Memory
{
std::shared_ptr<Chunk> ChunkType::allocate(vk::DeviceSize chunkSize)
{
    vk::MemoryAllocateInfo info(chunkSize, m_memoryTypeIndex);

    auto chunk =
        std::make_shared<Chunk>(SafeAllocator::instance()->allocateMemory(m_device, info), chunkSize, propertyFlags);
    m_chunks.push_back(chunk);
    return chunk;
}

void ChunkType::deallocate(std::shared_ptr<Chunk> chunk)
{
    m_device.free(chunk->memory);
    m_chunks.erase(std::remove(m_chunks.begin(), m_chunks.end(), chunk), m_chunks.end());
}
}  // namespace Saiga::Vulkan::Memory