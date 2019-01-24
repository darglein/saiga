//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once

#include "saiga/vulkan/memory/MemoryLocation.h"

#include <list>

namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
typedef std::vector<MemoryLocation> ChunkList;
typedef ChunkList::iterator LocationIterator;

struct SAIGA_GLOBAL ChunkAllocation
{
    std::shared_ptr<Chunk> chunk;
    vk::Buffer buffer;
    ChunkList allocations;
    ChunkList freeList;
    MemoryLocation maxFreeRange;
    void* mappedPointer;

    vk::DeviceSize allocated;
    vk::DeviceSize size;

    ChunkAllocation(const std::shared_ptr<Chunk>& _chunk, vk::Buffer _buffer, vk::DeviceSize _size,
                    void* _mappedPointer)
        : chunk(_chunk),
          buffer(_buffer),
          allocations(),
          freeList(),
          maxFreeRange(),
          mappedPointer(_mappedPointer),
          allocated(0),
          size(_size)
    {
        freeList.emplace_back(_buffer, _chunk->memory, 0, size);
        maxFreeRange = freeList.front();
    }

   public:
    bool operator<(const ChunkAllocation& rhs) const { return allocated > rhs.allocated; }

    inline vk::DeviceSize getFree() const { return size - allocated; }
};

typedef std::vector<ChunkAllocation>::iterator ChunkIterator;
typedef std::vector<ChunkAllocation>::reverse_iterator RevChunkIterator;

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga