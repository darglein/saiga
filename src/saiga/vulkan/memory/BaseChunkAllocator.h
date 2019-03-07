//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once


#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vulkan/Queue.h"
#include "saiga/vulkan/memory/MemoryStats.h"

#include "ChunkCreator.h"
#include "FitStrategy.h"
#include "MemoryStats.h"

#include <mutex>

namespace Saiga::Vulkan::Memory
{
template <typename T>
class SAIGA_VULKAN_API BaseChunkAllocator
{
   protected:
    std::mutex allocationMutex;
    void findNewMax(ChunkIterator<T>& chunkAlloc) const;
    vk::Device m_device;
    ChunkCreator* m_chunkAllocator{};

   public:
    FitStrategy<T>* strategy{};
    Queue* queue;

    vk::DeviceSize m_chunkSize{};
    vk::DeviceSize m_allocateSize{};
    ChunkContainer<T> chunks;

   protected:
    std::string gui_identifier;

    virtual ChunkIterator<T> createNewChunk() = 0;

    virtual void headerInfo() {}


    /**
     * Allocates \p size bytes
     * @tparam T Type of location to allocate
     * @param size Number of bytes to allocate
     * @remarks Function is not synchronized with a mutex. This must be done by the calling method.
     * @return A pointer to the allocated memory region. Data will not be set.
     */
    T* base_allocate(vk::DeviceSize size);

    virtual std::unique_ptr<T> create_location(ChunkIterator<T>& chunk_alloc, vk::DeviceSize start,
                                               vk::DeviceSize size) = 0;

   public:
    virtual void deallocate(T* location);
    BaseChunkAllocator(vk::Device _device, ChunkCreator* chunkAllocator, FitStrategy<T>& strategy, Queue* _queue,
                       vk::DeviceSize chunkSize = 64 * 1024 * 1024)
        : m_device(_device),
          m_chunkAllocator(chunkAllocator),
          strategy(&strategy),
          queue(_queue),
          m_chunkSize(chunkSize),
          m_allocateSize(chunkSize),
          gui_identifier("")
    {
    }

    BaseChunkAllocator(BaseChunkAllocator&& other) noexcept
        : m_device(other.m_device),
          m_chunkAllocator(other.m_chunkAllocator),
          strategy(other.strategy),
          queue(other.queue),
          m_chunkSize(other.m_chunkSize),
          m_allocateSize(other.m_allocateSize),
          chunks(std::move(other.chunks)),
          gui_identifier(std::move(other.gui_identifier))
    {
    }

    BaseChunkAllocator& operator=(BaseChunkAllocator&& other) noexcept
    {
        m_device         = other.m_device;
        m_chunkAllocator = other.m_chunkAllocator;
        strategy         = other.strategy;
        queue            = other.queue;
        m_chunkSize      = other.m_chunkSize;
        m_allocateSize   = other.m_allocateSize;
        chunks           = std::move(other.chunks);
        gui_identifier   = std::move(other.gui_identifier);
        return *this;
    }

    virtual ~BaseChunkAllocator() = default;

    T* reserve_space(vk::DeviceMemory memory, FreeListEntry freeListEntry, vk::DeviceSize size);


    bool memory_is_free(vk::DeviceMemory memory, FreeListEntry free_mem);

    void destroy();

    MemoryStats collectMemoryStats();

    void showDetailStats(bool expand);

    T* allocate_in_free_space(vk::DeviceSize size, ChunkIterator<T>& chunkAlloc, FreeIterator<T>& freeSpace);

    std::pair<ChunkIterator<T>, AllocationIterator<T>> find_allocation(T* location);

    void swap(T* target, T* source);

    template <typename FreeEntry>
    void add_to_free_list(const ChunkIterator<T>& chunk, const FreeEntry& location) const;
};

}  // namespace Saiga::Vulkan::Memory
