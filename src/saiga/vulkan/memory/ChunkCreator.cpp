//
// Created by Peter Eichinger on 08.10.18.
//

#include "ChunkCreator.h"

namespace Saiga::Vulkan::Memory
{
std::shared_ptr<Memory> ChunkType::allocate(vk::DeviceSize chunkSize)
{
    vk::MemoryAllocateInfo info(chunkSize, m_memoryTypeIndex);

    auto chunk =
        std::make_shared<Memory>(SafeAllocator::instance()->allocateMemory(m_device, info), chunkSize, propertyFlags);
    m_chunks.push_back(chunk);
    return chunk;
}

void ChunkType::deallocate(std::shared_ptr<Memory> chunk)
{
    m_device.free(chunk->memory);
    m_chunks.erase(std::remove(m_chunks.begin(), m_chunks.end(), chunk), m_chunks.end());
}

ChunkType& ChunkCreator::findMemoryType(vk::MemoryPropertyFlags flags)
{
    for (auto& memType : m_memoryTypes)
    {
        if ((memType.propertyFlags & flags) == flags)
        {
            return memType;
        }
    }
    throw std::runtime_error("No valid memory found");
}


void Saiga::Vulkan::Memory::ChunkCreator::init(vk::PhysicalDevice _physicalDevice, vk::Device _device)
{
    m_physicalDevice = _physicalDevice;
    m_device         = _device;

    auto memoryProperties = m_physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
        auto& properties = memoryProperties.memoryTypes[i];

        m_memoryTypes.emplace_back(m_device, i, properties.propertyFlags);
    }
    m_initialized = true;
}

std::shared_ptr<Memory> ChunkCreator::allocate(vk::MemoryPropertyFlags propertyFlags, vk::DeviceSize chunkSize)
{
    if (!m_initialized)
    {
        throw std::runtime_error("Must be initialized before use");
    }
    auto& memType = findMemoryType(propertyFlags);

    return memType.allocate(chunkSize);
}

void Saiga::Vulkan::Memory::ChunkCreator::deallocate(std::shared_ptr<Memory> chunk)
{
    if (!m_initialized)
    {
        throw std::runtime_error("Must be initialized before use");
    }

    findMemoryType(chunk->flags).deallocate(chunk);
}

vk::MemoryPropertyFlags Saiga::Vulkan::Memory::ChunkCreator::getEffectiveFlags(vk::MemoryPropertyFlags memoryFlags)
{
    return findMemoryType(memoryFlags).propertyFlags;
}
}  // namespace Saiga::Vulkan::Memory
