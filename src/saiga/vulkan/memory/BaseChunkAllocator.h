//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once


#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vulkan/Queue.h"
#include "saiga/vulkan/memory/BaseMemoryAllocator.h"

#include "BaseMemoryAllocator.h"
#include "ChunkCreator.h"
#include "FitStrategy.h"

#include <mutex>

namespace Saiga::Vulkan::Memory
{
template <typename T>
class SAIGA_VULKAN_API BaseChunkAllocator
{
   private:
    std::mutex allocationMutex;
    void findNewMax(ChunkIterator<T>& chunkAlloc) const;

   protected:
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


    virtual T* base_allocate(vk::DeviceSize size);

    virtual void base_deallocate(T* location);

    virtual std::unique_ptr<T> create_location(ChunkIterator<T>& chunk_alloc, vk::DeviceSize start,
                                               vk::DeviceSize size) = 0;

   public:
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


    bool memory_is_free(vk::DeviceMemory memory, FreeListEntry entry);

    void destroy();

    MemoryStats collectMemoryStats();

    void showDetailStats();

    T* allocate_in_free_space(vk::DeviceSize size, ChunkIterator<T>& chunkAlloc, FreeIterator<T>& freeSpace);

    std::pair<ChunkIterator<T>, AllocationIterator<T>> find_allocation(T* location);

    void move_allocation(T* target, T* source);

    void add_to_free_list(const ChunkIterator<T>& chunk, const T& location) const;
};

template <typename T>
T* BaseChunkAllocator<T>::base_allocate(vk::DeviceSize size)
{
    std::scoped_lock alloc_lock(allocationMutex);
    ChunkIterator<T> chunkAlloc;
    FreeIterator<T> freeSpace;
    std::tie(chunkAlloc, freeSpace) = strategy->findRange(chunks.begin(), chunks.end(), size);

    T* val = allocate_in_free_space(size, chunkAlloc, freeSpace);

    return val;
}

template <typename T>
T* BaseChunkAllocator<T>::allocate_in_free_space(vk::DeviceSize size, ChunkIterator<T>& chunkAlloc,
                                                 FreeIterator<T>& freeSpace)
{
    T* val;
    if (chunkAlloc == chunks.end())
    {
        chunkAlloc = createNewChunk();
        freeSpace  = chunkAlloc->freeList.begin();
    }

    auto memoryStart = freeSpace->offset;

    freeSpace->offset += size;
    freeSpace->size -= size;

    if (freeSpace->size == 0)
    {
        chunkAlloc->freeList.erase(freeSpace);
    }

    findNewMax(chunkAlloc);


    auto targetLocation = create_location(chunkAlloc, memoryStart, size);
    auto memoryEnd      = memoryStart + size;

    auto insertionPoint =
        lower_bound(chunkAlloc->allocations.begin(), chunkAlloc->allocations.end(), memoryEnd,
                    [](const auto& element, vk::DeviceSize value) { return element->offset < value; });

    val = chunkAlloc->allocations.insert(insertionPoint, move(targetLocation))->get();
    chunkAlloc->allocated += size;
    return val;
}

template <typename T>
void BaseChunkAllocator<T>::findNewMax(ChunkIterator<T>& chunkAlloc) const
{
    auto& freeList = chunkAlloc->freeList;

    if (chunkAlloc->freeList.empty())
    {
        chunkAlloc->maxFreeRange = std::nullopt;
        return;
    }


    chunkAlloc->maxFreeRange = *max_element(freeList.begin(), freeList.end(),
                                            [](auto& first, auto& second) { return first.size < second.size; });
}


template <typename T>
void BaseChunkAllocator<T>::base_deallocate(T* location)
{
    std::scoped_lock alloc_lock(allocationMutex);

    ChunkIterator<T> fChunk;
    AllocationIterator<T> fLoc;

    std::tie(fChunk, fLoc) = find_allocation(location);

    auto& chunkAllocs = fChunk->allocations;

    SAIGA_ASSERT(fLoc != chunkAllocs.end(), "Allocation is not part of the chunk");

    LOG(INFO) << "Deallocating " << location->size << " bytes in chunk/offset [" << distance(chunks.begin(), fChunk)
              << "/" << (*fLoc)->offset << "]";

    add_to_free_list(fChunk, *(fLoc->get()));

    findNewMax(fChunk);

    chunkAllocs.erase(fLoc);

    fChunk->allocated -= location->size;
    while (chunks.size() >= 2)
    {
        auto last = chunks.end() - 1;
        auto stol = chunks.end() - 2;

        if (!last->allocations.empty() || !stol->allocations.empty())
        {
            break;
        }

        m_device.destroy(last->buffer);
        m_chunkAllocator->deallocate(last->chunk);

        last--;
        stol--;

        chunks.erase(last + 1, chunks.end());
    }
}

template <typename T>
void BaseChunkAllocator<T>::destroy()
{
    for (auto& alloc : chunks)
    {
        m_device.destroy(alloc.buffer);
    }
}

template <typename T>
bool BaseChunkAllocator<T>::memory_is_free(vk::DeviceMemory memory, FreeListEntry free_mem)
{
    std::scoped_lock lock(allocationMutex);
    auto chunk = std::find_if(chunks.begin(), chunks.end(),
                              [&](const auto& chunk_entry) { return chunk_entry.chunk->memory == memory; });

    SAIGA_ASSERT(chunk != chunks.end(), "Wrong allocator");

    if (chunk->freeList.empty())
    {
        return false;
    }
    auto found =
        std::lower_bound(chunk->freeList.begin(), chunk->freeList.end(), free_mem,
                         [](const auto& free_entry, const auto& value) { return free_entry.offset < value.offset; });

    return found->offset == free_mem.offset && found->size == free_mem.size;
}


template <typename T>
T* BaseChunkAllocator<T>::reserve_space(vk::DeviceMemory memory, FreeListEntry freeListEntry, vk::DeviceSize size)
{
    std::scoped_lock lock(allocationMutex);
    auto chunk = std::find_if(chunks.begin(), chunks.end(),
                              [&](const auto& chunk_entry) { return chunk_entry.chunk->memory == memory; });

    SAIGA_ASSERT(chunk != chunks.end(), "Wrong allocator");

    auto free = std::find(chunk->freeList.begin(), chunk->freeList.end(), freeListEntry);

    SAIGA_ASSERT(free != chunk->freeList.end(), "Free space not found");

    return allocate_in_free_space(size, chunk, free);
}

template <typename T>
void BaseChunkAllocator<T>::move_allocation(T* target, T* source)
{
    // TODO: has to be done differently for images
    std::scoped_lock lock(allocationMutex);
    const auto size = source->size;

    T source_copy = *source;

    ChunkIterator<T> target_chunk, source_chunk;
    AllocationIterator<T> target_alloc, source_alloc;

    std::tie(target_chunk, target_alloc) = find_allocation(target);
    std::tie(source_chunk, source_alloc) = find_allocation(source);

    *source = *target;  // copy values from target to source;

    source->mark_dynamic();

    source_chunk->allocated -= size;

    std::move(source_alloc, std::next(source_alloc), target_alloc);

    source_chunk->allocations.erase(source_alloc);

    add_to_free_list(source_chunk, source_copy);


    findNewMax(source_chunk);
}

template <typename T>
void BaseChunkAllocator<T>::add_to_free_list(const ChunkIterator<T>& chunk, const T& location) const
{
    auto& freeList = chunk->freeList;
    auto found = lower_bound(freeList.begin(), freeList.end(), location, [](const auto& free_entry, const auto& value) {
        return free_entry.offset < value.offset;
    });

    FreeIterator<T> free;

    auto previous = prev(found);
    if (found != freeList.begin() && previous->end() == location.offset)
    {
        previous->size += location.size;
        free = previous;
    }
    else
    {
        free  = freeList.insert(found, FreeListEntry{location.offset, location.size});
        found = next(free);
    }


    if (found != freeList.end() && free->end() == found->offset)
    {
        free->size += found->size;
        freeList.erase(found);
    }
}

template <typename T>
std::pair<ChunkIterator<T>, AllocationIterator<T>> BaseChunkAllocator<T>::find_allocation(T* location)
{
    auto fChunk = std::find_if(chunks.begin(), chunks.end(), [&](ChunkAllocation<T> const& alloc) {
        return alloc.chunk->memory == location->memory;
    });

    SAIGA_ASSERT(fChunk != chunks.end(), "Allocation was not done with this allocator!");

    auto& chunkAllocs = fChunk->allocations;
    auto fLoc =
        std::lower_bound(chunkAllocs.begin(), chunkAllocs.end(), location,
                         [](const auto& element, const auto& value) { return element->offset < value->offset; });

    return std::make_pair(fChunk, fLoc);
}

template <typename T>
void BaseChunkAllocator<T>::showDetailStats()
{
    using BarColor = ImGui::ColoredBar::BarColor;
    static const BarColor alloc_color_static{{1.00f, 0.447f, 0.133f, 1.0f}, {0.133f, 0.40f, 0.40f, 1.0f}};
    static const BarColor alloc_color_dynamic{{1.00f, 0.812f, 0.133f, 1.0f}, {0.133f, 0.40f, 0.40f, 1.0f}};

    static std::vector<ImGui::ColoredBar> allocation_bars;

    if (ImGui::CollapsingHeader(gui_identifier.c_str()))
    {
        std::scoped_lock lock(allocationMutex);
        ImGui::Indent();

        headerInfo();

        allocation_bars.resize(
            chunks.size(), ImGui::ColoredBar({0, 40}, {{0.1f, 0.1f, 0.1f, 1.0f}, {0.4f, 0.4f, 0.4f, 1.0f}}, true, 1));

        int numAllocs           = 0;
        uint64_t usedSpace      = 0;
        uint64_t innerFreeSpace = 0;
        uint64_t totalFreeSpace = 0;
        for (auto i = 0U; i < allocation_bars.size(); ++i)
        {
            auto& bar   = allocation_bars[i];
            auto& chunk = chunks[i];

            std::stringstream ss;
            ss << "Mem " << std::hex << chunk.chunk->memory << " Buffer " << chunk.buffer;

            ImGui::Text("Chunk %d (%s, %s) %s", i + 1, sizeToString(chunk.getFree()).c_str(),
                        sizeToString(chunk.allocated).c_str(), ss.str().c_str());
            ImGui::Indent();
            bar.renderBackground();
            int j = 0;
            ConstAllocationIterator<T> allocIter;
            ConstFreeIterator<T> freeIter;
            for (allocIter = chunk.allocations.cbegin(), j = 0; allocIter != chunk.allocations.cend(); ++allocIter, ++j)
            {
                auto& color = (*allocIter)->is_static() ? alloc_color_static : alloc_color_dynamic;
                bar.renderArea(static_cast<float>((*allocIter)->offset) / m_chunkSize,
                               static_cast<float>((*allocIter)->offset + (*allocIter)->size) / m_chunkSize, color,
                               false);
                usedSpace += (*allocIter)->size;
            }
            numAllocs += j;

            if (!chunk.freeList.empty())
            {
                auto freeEnd = --chunk.freeList.cend();
                for (freeIter = chunk.freeList.cbegin(); freeIter != freeEnd; freeIter++)
                {
                    innerFreeSpace += freeIter->size;
                    totalFreeSpace += freeIter->size;
                }

                totalFreeSpace += chunk.freeList.back().size;
            }
            ImGui::Unindent();
        }
        ImGui::LabelText("Number of allocations", "%d", numAllocs);
        auto totalSpace = m_chunkSize * chunks.size();


        ImGui::LabelText("Usage", "%s / %s (%.2f%%)", sizeToString(usedSpace).c_str(), sizeToString(totalSpace).c_str(),
                         100 * static_cast<float>(usedSpace) / totalSpace);
        ImGui::LabelText("Free Space (total / fragmented)", "%s / %s", sizeToString(totalFreeSpace).c_str(),
                         sizeToString(innerFreeSpace).c_str());


        ImGui::Unindent();
    }
}

template <typename T>
MemoryStats BaseChunkAllocator<T>::collectMemoryStats()
{
    std::scoped_lock lock(allocationMutex);
    int numAllocs                = 0;
    uint64_t usedSpace           = 0;
    uint64_t fragmentedFreeSpace = 0;
    uint64_t totalFreeSpace      = 0;
    for (auto& chunk : chunks)
    {
        int j = 0;
        ConstAllocationIterator<T> allocIter;
        ConstFreeIterator<T> freeIter;
        for (allocIter = chunk.allocations.cbegin(), j = 0; allocIter != chunk.allocations.cend(); ++allocIter, ++j)
        {
            usedSpace += (*allocIter)->size;
        }
        numAllocs += j;

        if (!chunk.freeList.empty())
        {
            auto freeEnd = --chunk.freeList.cend();

            for (freeIter = chunk.freeList.cbegin(); freeIter != freeEnd; freeIter++)
            {
                fragmentedFreeSpace += freeIter->size;
                totalFreeSpace += freeIter->size;
            }

            totalFreeSpace += chunk.freeList.back().size;
        }
    }
    auto totalSpace = m_chunkSize * chunks.size();
    //
    return MemoryStats{totalSpace, usedSpace, fragmentedFreeSpace};
}

}  // namespace Saiga::Vulkan::Memory
