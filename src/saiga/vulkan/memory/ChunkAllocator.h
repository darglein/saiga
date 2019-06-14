//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once


#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vulkan/Queue.h"
#include "saiga/vulkan/memory/MemoryStats.h"

#include "FitStrategy.h"
#include "MemoryStats.h"

#include <mutex>

namespace Saiga::Vulkan::Memory
{
template <typename T>
class SAIGA_VULKAN_API ChunkAllocator
{
   protected:
    std::mutex allocationMutex;
    void findNewMax(ChunkIterator<T>& chunkAlloc) const;
    vk::PhysicalDevice m_pDevice;
    vk::Device m_device;


   public:
    double totalTime = 0.0;
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

   private:
    T* allocate_in_free_space(vk::DeviceSize size, ChunkIterator<T>& chunkAlloc, FreeIterator<T>& freeSpace);

    template <typename FreeEntry>
    void add_to_free_list(const ChunkIterator<T>& chunk, const FreeEntry& location) const;

   public:
    virtual void deallocate(T* location);
    ChunkAllocator(vk::PhysicalDevice _pDevice, vk::Device _device, FitStrategy<T>& strategy, Queue* _queue,
                   vk::DeviceSize chunkSize = 64 * 1024 * 1024)
        : m_pDevice(_pDevice),
          m_device(_device),
          strategy(&strategy),
          queue(_queue),
          m_chunkSize(chunkSize),
          m_allocateSize(chunkSize),
          gui_identifier("")
    {
    }

    ChunkAllocator(ChunkAllocator&& other) noexcept
        : m_pDevice(other.m_pDevice),
          m_device(other.m_device),
          strategy(other.strategy),
          queue(other.queue),
          m_chunkSize(other.m_chunkSize),
          m_allocateSize(other.m_allocateSize),
          chunks(std::move(other.chunks)),
          gui_identifier(std::move(other.gui_identifier))
    {
    }

    ChunkAllocator& operator=(ChunkAllocator&& other) noexcept
    {
        m_pDevice      = other.m_pDevice;
        m_device       = other.m_device;
        strategy       = other.strategy;
        queue          = other.queue;
        m_chunkSize    = other.m_chunkSize;
        m_allocateSize = other.m_allocateSize;
        chunks         = std::move(other.chunks);
        gui_identifier = std::move(other.gui_identifier);
        return *this;
    }


    virtual ~ChunkAllocator() { destroy(); }

    ChunkAllocator(const ChunkAllocator&) = delete;
    ChunkAllocator& operator=(const ChunkAllocator&) = delete;

    T* reserve_if_free(vk::DeviceMemory memory, FreeListEntry freeListEntry, vk::DeviceSize size);

    void destroy();

    MemoryStats collectMemoryStats();

    void showDetailStats(bool expand);

    std::pair<ChunkIterator<T>, AllocationIterator<T>> find_allocation(T* location);

    void swap(T* target, T* source);

    inline bool can_allocate(vk::DeviceSize size) const { return size <= m_chunkSize; }
};

}  // namespace Saiga::Vulkan::Memory
