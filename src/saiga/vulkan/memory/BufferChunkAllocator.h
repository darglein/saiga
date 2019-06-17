#include <utility>

//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include "saiga/core/math/imath.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/export.h"

#include "BufferMemoryLocation.h"
#include "Chunk.h"
#include "ChunkAllocator.h"
#include "FitStrategy.h"
#include "MemoryStats.h"
#include "MemoryType.h"

#include <limits>
#include <list>
#include <utility>
#include <vulkan/vulkan.hpp>

namespace Saiga::Vulkan::Memory
{
class SAIGA_VULKAN_API BufferChunkAllocator final : public ChunkAllocator<BufferMemoryLocation>
{
   private:
    vk::DeviceSize m_alignment = std::numeric_limits<vk::DeviceSize>::max();
    vk::BufferCreateInfo m_bufferCreateInfo;

   protected:
    ChunkIterator<BufferMemoryLocation> createNewChunk() override;

    void headerInfo() override;

    std::unique_ptr<BufferMemoryLocation> create_location(ChunkIterator<BufferMemoryLocation>& chunk_alloc,
                                                          vk::DeviceSize start, vk::DeviceSize size) override;

   public:
    BufferType type;

    BufferChunkAllocator(vk::PhysicalDevice _pDevice, vk::Device _device, BufferType _type,
                         FitStrategy<BufferMemoryLocation>& strategy, Queue* _queue,
                         vk::DeviceSize chunkSize = 64 * 1024 * 1024)
        : ChunkAllocator(_pDevice, _device, strategy, _queue, chunkSize), type(std::move(_type))
    {
        std::stringstream identifier_stream;
        identifier_stream << "Buffer Chunk " << type;
        gui_identifier                 = identifier_stream.str();
        m_bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        m_bufferCreateInfo.usage       = type.usageFlags;
        m_bufferCreateInfo.size        = m_chunkSize;
        auto buffer                    = m_device.createBuffer(m_bufferCreateInfo);
        auto requirements              = m_device.getBufferMemoryRequirements(buffer);
        m_allocateSize                 = requirements.size;
        m_alignment                    = requirements.alignment;
        m_device.destroy(buffer);

        VLOG(3) << "Created new buffer allocator  " << type << ", alignment " << m_alignment;
    }

    BufferChunkAllocator(BufferChunkAllocator&& other) noexcept
        : ChunkAllocator(std::move(other)),
          m_alignment(other.m_alignment),
          m_bufferCreateInfo(std::move(other.m_bufferCreateInfo))
    {
    }


    BufferChunkAllocator& operator=(BufferChunkAllocator&& other) noexcept
    {
        ChunkAllocator::operator=(std::move(static_cast<ChunkAllocator&&>(other)));
        m_alignment             = other.m_alignment;
        m_bufferCreateInfo      = other.m_bufferCreateInfo;
        return *this;
    }


    void deallocate(BufferMemoryLocation* location) override;

    BufferChunkAllocator(const BufferChunkAllocator&) = delete;

    BufferChunkAllocator& operator=(const BufferChunkAllocator&) = delete;

    BufferMemoryLocation* allocate(vk::DeviceSize size);
};
}  // namespace Saiga::Vulkan::Memory
