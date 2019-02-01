//
// Created by Peter Eichinger on 08.10.18.
//

#pragma once
#include "saiga/export.h"

#include "Chunk.h"

#include <vector>
#include <vulkan/vulkan.hpp>
namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
/**
 * Abstraction for a single memory type as provided by vkGetPhysicalDeviceMemoryProperties().
 * Allows allocation an deallocation of memory in chunks of the size chunkSize.
 */
class SAIGA_VULKAN_API ChunkType
{
   private:
    uint32_t m_memoryTypeIndex;
    const vk::Device& m_device;
    std::vector<std::shared_ptr<Chunk>> m_chunks;

   public:
    vk::MemoryPropertyFlags propertyFlags;

    ChunkType(const ChunkType& memoryType) = delete;

    ChunkType(ChunkType&& memoryType) noexcept
        : m_memoryTypeIndex(memoryType.m_memoryTypeIndex),
          m_device(memoryType.m_device),
          m_chunks(std::move(memoryType.m_chunks)),
          propertyFlags(memoryType.propertyFlags)
    {
    }


    ChunkType(const vk::Device& device, uint32_t memoryTypeIndex, vk::MemoryPropertyFlags flags)
        : m_memoryTypeIndex(memoryTypeIndex), m_device(device), propertyFlags(flags)
    {
    }

    std::shared_ptr<Chunk> allocate(vk::DeviceSize chunkSize);

    void deallocate(std::shared_ptr<Chunk> chunk);

    void destroy()
    {
        for (auto& chunk : m_chunks)
        {
            m_device.free(chunk->memory);
        }
        m_chunks.clear();
    }
};


}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
