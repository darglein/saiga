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
#include <list>
#include <utility>
using namespace Saiga::Vulkan::Memory;

namespace Saiga{
namespace Vulkan{
namespace Memory{

struct SAIGA_GLOBAL MemoryRange {
    vk::DeviceSize start;
    vk::DeviceSize range;
    MemoryRange(vk::DeviceSize _start, vk::DeviceSize _range) : start(_start), range(_range) { }
};

struct SAIGA_GLOBAL ChunkAllocation{
    std::shared_ptr<Chunk> chunk;
    vk::Buffer buffer;
    std::list<MemoryRange> allocations;
    std::list<MemoryRange> freeList;
    vk::DeviceSize maxFreeSize;
    MemoryRange* maxFreeRange;

    ChunkAllocation(std::shared_ptr<Chunk> _chunk, vk::Buffer _buffer, vk::DeviceSize size) {
        chunk = _chunk;
        buffer = _buffer;
        freeList.emplace_back(MemoryRange{0, size});
        maxFreeSize = size;
        maxFreeRange = &(*freeList.begin());
    }
};

struct SAIGA_GLOBAL FitStrategy {
    virtual std::pair<ChunkAllocation*, MemoryRange*> findRange(std::vector<ChunkAllocation> const & _allocations, vk::DeviceSize size) = 0;
};



class SAIGA_GLOBAL ChunkMemoryAllocator : public MemoryAllocatorBase {
private:
    vk::Device m_device;
    ChunkAllocator* m_chunkAllocator;
    vk::DeviceSize m_chunkSize;

    vk::DeviceSize m_allocateSize;
    vk::DeviceSize m_alignment = std::numeric_limits<vk::DeviceSize>::max();
    vk::BufferCreateInfo m_bufferCreateInfo;

    std::string m_logger;

    std::vector<ChunkAllocation> m_chunkAllocations;
    std::shared_ptr<FitStrategy> m_strategy;

    /**
     * Create a new chunk. Returns a pointer to the free entry of the new chunk.
     * @return Pointer to the free entry of the new chunk.
     */
    std::pair<ChunkAllocation*, MemoryRange*> createNewChunk();
public:
    vk::MemoryPropertyFlags flags;
    vk::BufferUsageFlags usageFlags;

    void init(vk::Device _device, ChunkAllocator* chunkAllocator, const vk::MemoryPropertyFlags &_flags,
                    const vk::BufferUsageFlags &usage, vk::DeviceSize chunkSize = 64* 1024* 1024, const std::string& name = "");


    MemoryLocation allocate(vk::DeviceSize size) override;

    void destroy();
};
}
}
}


