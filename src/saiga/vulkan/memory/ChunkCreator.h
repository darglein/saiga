//
// Created by Peter Eichinger on 08.10.18.
//

#pragma once

#include "MemoryLocation.h"
#include "SafeAllocator.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace Saiga::Vulkan::Memory
{
struct Memory
{
    vk::DeviceMemory memory;
    vk::DeviceSize size;
    vk::MemoryPropertyFlags flags;

    Memory(vk::DeviceMemory _memory, vk::DeviceSize _size, vk::MemoryPropertyFlags _flags)
        : memory(_memory), size(_size), flags(_flags)
    {
    }
};

/**
 * Abstraction for a single memory type as provided by vkGetPhysicalDeviceMemoryProperties().
 * Allows allocation an deallocation of memory in chunks of the size chunkSize.
 */
class SAIGA_VULKAN_API ChunkType
{
   private:
    uint32_t m_memoryTypeIndex;
    const vk::Device& m_device;
    std::vector<std::shared_ptr<Memory>> m_chunks;

   public:
    vk::MemoryPropertyFlags propertyFlags;

    ChunkType(const ChunkType& memoryType) = delete;
    ChunkType& operator=(const ChunkType&) = delete;
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

    std::shared_ptr<Memory> allocate(vk::DeviceSize chunkSize);

    void deallocate(std::shared_ptr<Memory> chunk);

    void destroy()
    {
        for (auto& chunk : m_chunks)
        {
            m_device.free(chunk->memory);
        }
        m_chunks.clear();
    }
};

/**
 * Class that allocates chunks of memory for different types.
 * Returns shared pointers to the chunks.
 */
class SAIGA_VULKAN_API ChunkCreator
{
   private:
    bool m_initialized = false;
    vk::PhysicalDevice m_physicalDevice;
    vk::Device m_device;
    std::vector<ChunkType> m_memoryTypes;

    ChunkType& findMemoryType(vk::MemoryPropertyFlags flags);

   public:
    ChunkCreator() {}

    ChunkCreator(const ChunkCreator&) = delete;
    ChunkCreator& operator=(const ChunkCreator&) = delete;

    ChunkCreator(ChunkCreator&& other) = default;
    ChunkCreator& operator=(ChunkCreator&& other) = default;

    void init(vk::PhysicalDevice _physicalDevice, vk::Device _device);

    vk::MemoryPropertyFlags getEffectiveFlags(vk::MemoryPropertyFlags memoryFlags);

    std::shared_ptr<Memory> allocate(vk::MemoryPropertyFlags propertyFlags, vk::DeviceSize chunkSize);

    void deallocate(std::shared_ptr<Memory> chunk);

    void destroy()
    {
        for (auto& type : m_memoryTypes)
        {
            type.destroy();
        }
    }
};

}  // namespace Saiga::Vulkan::Memory
