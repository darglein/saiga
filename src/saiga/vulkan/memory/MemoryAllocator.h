//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include <vulkan/vulkan.hpp>
#include "saiga/export.h"
#include "saiga/vulkan/memory/ChunkAllocator.h"
#include "saiga/util/imath.h"
#include "saiga/vulkan/memory/MemoryLocation.h"
#include "saiga/vulkan/memory/MemoryAllocatorBase.h"
#include <limits>
using namespace Saiga::Vulkan::Memory;

namespace Saiga{
namespace Vulkan{
namespace Memory{

struct SAIGA_GLOBAL MemoryAllocator : public MemoryAllocatorBase {
private:
    ChunkAllocator* m_chunkAllocator;
    vk::Device m_device;
    vk::Buffer m_currentBuffer;
    std::shared_ptr<MemoryChunk> m_currentChunk = nullptr;
    vk::DeviceSize m_currentOffset = 0;

    vk::DeviceSize m_chunkSize;
    vk::DeviceSize m_allocateSize;

    vk::BufferCreateInfo m_bufferCreateInfo;


    vk::DeviceSize m_alignment = std::numeric_limits<vk::DeviceSize>::max();

    void createNewBuffer() {
        m_currentChunk = m_chunkAllocator->allocate(flags, m_allocateSize);
        m_currentBuffer = m_device.createBuffer(m_bufferCreateInfo);
        m_currentOffset = 0;
        m_device.getBufferMemoryRequirements(m_currentBuffer);
        m_device.bindBufferMemory(m_currentBuffer, m_currentChunk->memory, 0);
    }
public:
    vk::MemoryPropertyFlags flags;
    vk::BufferUsageFlags usageFlags;

    void init(vk::Device _device, ChunkAllocator* chunkAllocator, const vk::MemoryPropertyFlags &_flags,
                    const vk::BufferUsageFlags &usage, vk::DeviceSize chunkSize = 64* 1024* 1024) {
        m_device = _device;
        m_chunkAllocator = chunkAllocator;
        m_chunkSize= chunkSize;
        flags = _flags;
        usageFlags = usage;
        m_bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        m_bufferCreateInfo.usage = usage;
        m_bufferCreateInfo.size = m_chunkSize;

        auto buffer = m_device.createBuffer(m_bufferCreateInfo);
        auto requirements = m_device.getBufferMemoryRequirements(buffer);
        m_allocateSize = requirements.size;
        m_alignment = requirements.alignment;

        if (m_allocateSize != m_chunkSize) {
            std::cerr << vk::to_string(usage) <<  " buffer usage: Allocation / Chunk size is different!!!: " << m_allocateSize << "/" << m_chunkSize << std::endl;
        }
        m_device.destroy(buffer);
    }


    MemoryLocation allocate(vk::DeviceSize size) override {
        if (m_currentChunk == nullptr) {
            createNewBuffer();
        }

        auto alignedSize = iAlignUp(size, m_alignment);

        if (m_currentOffset + alignedSize > m_chunkSize) {
            createNewBuffer();
        }

        MemoryLocation targetLocation = {m_currentBuffer, m_currentChunk->memory, m_currentOffset,size};
        m_currentOffset += alignedSize;
        return targetLocation;
    }
};
}
}
}


