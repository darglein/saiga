//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include <limits>
#include <list>
#include <utility>
#include <vulkan/vulkan.hpp>
#include "BaseChunkAllocator.h"
#include "saiga/export.h"
#include "saiga/util/easylogging++.h"
#include "saiga/util/imath.h"
#include "saiga/vulkan/memory/BaseMemoryAllocator.h"
#include "saiga/vulkan/memory/ChunkAllocation.h"
#include "saiga/vulkan/memory/ChunkCreator.h"
#include "saiga/vulkan/memory/FitStrategy.h"
#include "saiga/vulkan/memory/MemoryLocation.h"

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
    ~BufferChunkAllocator() override = default;
    vk::BufferUsageFlags usageFlags;

    BufferChunkAllocator(vk::Device _device, ChunkCreator* chunkAllocator, const vk::MemoryPropertyFlags& _flags,
                         const vk::BufferUsageFlags& usage, FitStrategy& strategy,
                         vk::DeviceSize chunkSize = 64 * 1024 * 1024, bool _mapped = false)
        : BaseChunkAllocator(_device, chunkAllocator, _flags, strategy, chunkSize, _mapped), usageFlags(usage)
    {
        m_bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        m_bufferCreateInfo.usage       = usageFlags;
        m_bufferCreateInfo.size        = m_chunkSize;
        auto buffer                    = m_device.createBuffer(m_bufferCreateInfo);
        auto requirements              = m_device.getBufferMemoryRequirements(buffer);
        m_allocateSize                 = requirements.size;
        m_alignment                    = requirements.alignment;
        m_device.destroy(buffer);

        LOG(INFO) << "Created new buffer allocator for " << vk::to_string(usageFlags) << ", MemType "
                  << vk::to_string(flags) << ", alignment " << m_alignment;
    }

    BufferChunkAllocator(BufferChunkAllocator&& other) noexcept
        : BaseChunkAllocator(std::move(other)),
          m_alignment(other.m_alignment),
          m_bufferCreateInfo(std::move(other.m_bufferCreateInfo))
    {
    }


    BufferChunkAllocator& operator=(BufferChunkAllocator&& other) noexcept
    {
        BaseChunkAllocator::operator=(std::move(other));
        m_alignment                 = other.m_alignment;
        m_bufferCreateInfo          = other.m_bufferCreateInfo;
        return *this;
    }

    void deallocate(MemoryLocation& location) override;

    MemoryLocation allocate(vk::DeviceSize size) override;
};
}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
