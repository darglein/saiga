#include <utility>

//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include "saiga/export.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/imath.h"

#include "BaseChunkAllocator.h"
#include "BaseMemoryAllocator.h"
#include "ChunkAllocation.h"
#include "ChunkCreator.h"
#include "FitStrategy.h"
#include "MemoryLocation.h"
#include "MemoryType.h"

#include <limits>
#include <list>
#include <utility>
#include <vulkan/vulkan.hpp>
using namespace Saiga::Vulkan::Memory;

namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
class SAIGA_GLOBAL BufferChunkAllocator : public BaseChunkAllocator
{
   private:
    vk::DeviceSize m_alignment = std::numeric_limits<vk::DeviceSize>::max();
    vk::BufferCreateInfo m_bufferCreateInfo;

   protected:
    ChunkIterator createNewChunk() override;

   public:
    BufferType type;
    ~BufferChunkAllocator() override = default;

    BufferChunkAllocator(vk::Device _device, ChunkCreator* chunkAllocator, BufferType _type, FitStrategy& strategy,
                         vk::DeviceSize chunkSize = 64 * 1024 * 1024)
        : BaseChunkAllocator(_device, chunkAllocator, strategy, chunkSize), type(std::move(_type))
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

        LOG(INFO) << "Created new buffer allocator  " << type << ", alignment " << m_alignment;
    }

    BufferChunkAllocator(BufferChunkAllocator&& other) noexcept
        : BaseChunkAllocator(std::move(other)),
          m_alignment(other.m_alignment),
          m_bufferCreateInfo(std::move(other.m_bufferCreateInfo))
    {
    }


    BufferChunkAllocator& operator=(BufferChunkAllocator&& other) noexcept
    {
        BaseChunkAllocator::operator=(std::move(static_cast<BaseChunkAllocator&&>(other)));
        m_alignment                 = other.m_alignment;
        m_bufferCreateInfo          = other.m_bufferCreateInfo;
        return *this;
    }

    void deallocate(MemoryLocation& location) override;

    MemoryLocation allocate(vk::DeviceSize size) override;

   protected:
    void headerInfo() override;
};
}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
