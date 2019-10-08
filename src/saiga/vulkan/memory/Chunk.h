#include <utility>

//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once

#include "MemoryLocation.h"

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

    inline vk::DeviceSize end() const { return offset + size; }
};



template <typename T>
struct SAIGA_TEMPLATE Chunk
{
    using FreeList      = std::vector<FreeListEntry>;
    using FreeIterator  = typename std::vector<FreeListEntry>::iterator;
    using AllocatedList = std::vector<std::unique_ptr<T>>;

    vk::DeviceMemory memory;
    vk::Buffer buffer;
    AllocatedList allocations;
    FreeList freeList;
    std::optional<FreeListEntry> maxFreeRange;
    void* mappedPointer;

    vk::DeviceSize allocated;
    vk::DeviceSize size;
    Chunk() = default;

    Chunk(Chunk&&) = default;
    Chunk& operator=(Chunk&&) = default;
    Chunk(const Chunk&)       = delete;
    Chunk& operator=(const Chunk&) = delete;

    Chunk(vk::DeviceMemory _memory, vk::Buffer _buffer, vk::DeviceSize _size, void* _mappedPointer)
        : memory(_memory),
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
    float getFragmentation() const
    {
        auto max = 0U;
        auto sum = 0U;
        for (auto& freeEntry : freeList)
        {
            sum += freeEntry.size;
            if (max < freeEntry.size)
            {
                max = freeEntry.size;
            }
        }

        if (sum == 0U)
        {
            return 0.0f;
        }

        return 1.0f - (static_cast<float>(max) / sum);
    }
    inline vk::DeviceSize getFree() const { return size - allocated; }
};

template <typename T>
using ChunkContainer = std::vector<Chunk<T>>;

template <typename T>
using ChunkIterator = typename ChunkContainer<T>::iterator;

template <typename T>
using ConstChunkIterator = typename ChunkContainer<T>::const_iterator;

template <typename T>
using RevChunkIterator = typename ChunkContainer<T>::reverse_iterator;

template <typename T>
using ConstRevChunkIterator = typename ChunkContainer<T>::const_reverse_iterator;

template <typename T>
using AllocationIterator = typename Chunk<T>::AllocatedList::iterator;

template <typename T>
using ConstAllocationIterator = typename Chunk<T>::AllocatedList::const_iterator;

template <typename T>
using RevAllocationIterator = typename Chunk<T>::AllocatedList::reverse_iterator;

template <typename T>
using ConstRevAllocationIterator = typename Chunk<T>::AllocatedList::const_reverse_iterator;

template <typename T>
using FreeIterator = typename Chunk<T>::FreeList::iterator;

template <typename T>
using ConstFreeIterator = typename Chunk<T>::FreeList::const_iterator;
}  // namespace Saiga::Vulkan::Memory
