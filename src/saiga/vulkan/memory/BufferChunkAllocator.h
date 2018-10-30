//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include <vulkan/vulkan.hpp>
#include "saiga/export.h"
#include "saiga/vulkan/memory/ChunkBuilder.h"
#include "saiga/util/imath.h"
#include "saiga/vulkan/memory/MemoryLocation.h"
#include "saiga/vulkan/memory/BaseMemoryAllocator.h"
#include <limits>
#include <list>
#include <utility>
#include "saiga/util/easylogging++.h"
#include "saiga/vulkan/memory/FitStrategy.h"
#include "saiga/vulkan/memory/ChunkAllocation.h"
#include "BaseChunkAllocator.h"

using namespace Saiga::Vulkan::Memory;

namespace Saiga{
namespace Vulkan{
namespace Memory{


class SAIGA_GLOBAL BufferChunkAllocator : public BaseChunkAllocator {
private:
    vk::DeviceSize m_alignment = std::numeric_limits<vk::DeviceSize>::max();
    vk::BufferCreateInfo m_bufferCreateInfo;

protected:
    ChunkIterator createNewChunk() override;

public:
    vk::BufferUsageFlags usageFlags;

    BufferChunkAllocator() : BaseChunkAllocator() {}
    BufferChunkAllocator(vk::Device _device, ChunkBuilder* chunkAllocator, const vk::MemoryPropertyFlags &_flags,
                    const vk::BufferUsageFlags &usage, FitStrategy& strategy, vk::DeviceSize chunkSize = 64* 1024* 1024,
                    bool _mapped = false) : BaseChunkAllocator(_device, chunkAllocator, _flags, strategy, chunkSize,_mapped),
                    usageFlags(usage) {
        m_bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        m_bufferCreateInfo.usage = usageFlags;
        m_bufferCreateInfo.size = m_chunkSize;
        auto buffer = m_device.createBuffer(m_bufferCreateInfo);
        auto requirements = m_device.getBufferMemoryRequirements(buffer);
        m_allocateSize = requirements.size;
        m_alignment = requirements.alignment;
        m_device.destroy(buffer);

        LOG(INFO) << "Created new buffer allocator for "<< vk::to_string(usageFlags)<< ", MemType " << vk::to_string(flags) << ", alignment " << m_alignment;
    }


    MemoryLocation allocate(vk::DeviceSize size) override;

    void destroy();

};
}
}
}


