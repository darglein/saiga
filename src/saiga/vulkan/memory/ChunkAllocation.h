#include <utility>

//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once

#include "saiga/vulkan/memory/MemoryLocation.h"

#include <list>
#include <memory>
#include <optional>
#include <ostream>
namespace Saiga::Vulkan::Memory
{
struct FreeListEntry
{
    vk::DeviceSize offset;
    vk::DeviceSize size;
    FreeListEntry() : offset(VK_WHOLE_SIZE), size(0) {}
    FreeListEntry(vk::DeviceSize _offset, vk::DeviceSize _size) : offset(_offset), size(_size) {}

    bool operator==(const FreeListEntry& rhs) const { return std::tie(offset, size) == std::tie(rhs.offset, rhs.size); }

    bool operator!=(const FreeListEntry& rhs) const { return !(rhs == *this); }

    friend std::ostream& operator<<(std::ostream& os, const FreeListEntry& entry)
    {
        os << "offset: " << entry.offset << " size: " << entry.size;
        return os;
    }

    inline vk::DeviceSize end() { return offset + size; }
};

using FreeList          = std::vector<FreeListEntry>;
using FreeIterator      = FreeList::iterator;
using ConstFreeIterator = FreeList::const_iterator;

using AllocatedList           = std::vector<std::unique_ptr<MemoryLocation>>;
using AllocationIterator      = AllocatedList::iterator;
using ConstAllocationIterator = AllocatedList::const_iterator;

struct SAIGA_VULKAN_API ChunkAllocation
{
    std::shared_ptr<Chunk> chunk;
    vk::Buffer buffer;
    AllocatedList allocations;
    FreeList freeList;
    std::optional<FreeListEntry> maxFreeRange;
    void* mappedPointer;

    vk::DeviceSize allocated;
    vk::DeviceSize size;

    ChunkAllocation(std::shared_ptr<Chunk> _chunk, vk::Buffer _buffer, vk::DeviceSize _size, void* _mappedPointer)
        : chunk(std::move(_chunk)),
          buffer(_buffer),
          allocations(),
          freeList(),
          maxFreeRange(),
          mappedPointer(_mappedPointer),
          allocated(0),
          size(_size)
    {
        freeList.emplace_back(0, size);
        maxFreeRange = freeList.front();
    }

   public:
    inline vk::DeviceSize getFree() const { return size - allocated; }
};


typedef std::vector<ChunkAllocation> ChunkContainer;
typedef ChunkContainer::iterator ChunkIterator;
typedef ChunkContainer::const_iterator ConstChunkIterator;
typedef ChunkContainer::reverse_iterator RevChunkIterator;
typedef ChunkContainer::const_reverse_iterator ConstRevChunkIterator;

}  // namespace Saiga::Vulkan::Memory