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

//struct SAIGA_GLOBAL MemoryRange {
//    vk::DeviceSize start;
//    vk::DeviceSize size;
//    MemoryRange(vk::DeviceSize _start, vk::DeviceSize _size) : start(_start), size(_size) { }
//};

struct SAIGA_GLOBAL ChunkAllocation{
    std::shared_ptr<Chunk> chunk;
    vk::Buffer buffer;
    std::list<MemoryLocation> allocations;
    std::list<MemoryLocation> freeList;
    vk::DeviceSize maxFreeSize;
    MemoryLocation* maxFreeRange;

    ChunkAllocation(std::shared_ptr<Chunk> _chunk, vk::Buffer _buffer, vk::DeviceSize size) {
        chunk = _chunk;
        buffer = _buffer;
        freeList.emplace_back(_buffer, _chunk->memory, 0, size);
        maxFreeSize = size;
        maxFreeRange = &freeList.front();
    }
};

typedef std::vector<ChunkAllocation>::iterator ChunkIterator;
typedef std::list<MemoryLocation>::iterator LocationIterator;
struct SAIGA_GLOBAL FitStrategy {
    virtual std::pair<ChunkIterator, LocationIterator> findRange(std::vector<ChunkAllocation> & _allocations, vk::DeviceSize size) = 0;
};

struct SAIGA_GLOBAL FirstFitStrategy : public FitStrategy{
    std::pair<ChunkIterator, LocationIterator>
    findRange(std::vector<ChunkAllocation> &_allocations, vk::DeviceSize size) override;
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
     * Create a new chunk.
     * @return Iterator to the new chunk.
     */
    ChunkIterator createNewChunk();
public:
    vk::MemoryPropertyFlags flags;
    vk::BufferUsageFlags usageFlags;

    void init(vk::Device _device, ChunkAllocator* chunkAllocator, const vk::MemoryPropertyFlags &_flags,
                    const vk::BufferUsageFlags &usage, std::shared_ptr<FitStrategy> strategy, vk::DeviceSize chunkSize = 64* 1024* 1024, const std::string& name = "");


    MemoryLocation& allocate(vk::DeviceSize size) override;

    void deallocate(MemoryLocation &location) override;

    void destroy();
};
}
}
}


