//
// Created by Peter Eichinger on 2019-03-07.
//
#include "ChunkAllocator.h"

#include "BufferMemoryLocation.h"
#include "ImageMemoryLocation.h"

namespace Saiga::Vulkan::Memory
{
template <typename T>
T* ChunkAllocator<T>::base_allocate(vk::DeviceSize size)
{
    ChunkIterator<T> chunkAlloc;
    FreeIterator<T> freeSpace;

    double time;
    {
        auto timer                      = make_scoped_timer<std::chrono::microseconds>(time);
        std::tie(chunkAlloc, freeSpace) = strategy->findRange(chunks.begin(), chunks.end(), size);
    }
    totalTime += time;
    T* val = allocate_in_free_space(size, chunkAlloc, freeSpace);

    return val;
}

template <typename T>
T* ChunkAllocator<T>::allocate_in_free_space(vk::DeviceSize size, ChunkIterator<T>& chunkAlloc,
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
void ChunkAllocator<T>::findNewMax(ChunkIterator<T>& chunkAlloc) const
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
void ChunkAllocator<T>::deallocate(T* location)
{
    std::scoped_lock alloc_lock(allocationMutex);

    ChunkIterator<T> fChunk;
    AllocationIterator<T> fLoc;

    std::tie(fChunk, fLoc) = find_allocation(location);

    auto& chunkAllocs = fChunk->allocations;

    SAIGA_ASSERT(fLoc != chunkAllocs.end(), "Allocation is not part of the chunk");

    (**fLoc).destroy_owned_data(m_device);
    VLOG(3) << "Deallocating " << location->size << " bytes in chunk/offset [" << distance(chunks.begin(), fChunk)
            << "/" << (*fLoc)->offset << "]";

    fChunk->allocated -= location->size;
    add_to_free_list(fChunk, *(fLoc->get()));

    findNewMax(fChunk);

    chunkAllocs.erase(fLoc);

    while (chunks.size() >= 2)
    {
        auto last = chunks.end() - 1;
        auto stol = chunks.end() - 2;

        if (!last->allocations.empty() || !stol->allocations.empty())
        {
            break;
        }

        m_device.destroy(last->buffer);
        m_device.free(last->memory);

        chunks.erase(last);
    }
}

template <typename T>
void ChunkAllocator<T>::destroy()
{
    for (auto& alloc : chunks)
    {
        m_device.destroy(alloc.buffer);
        m_device.free(alloc.memory);
    }
    chunks.clear();
}

template <typename T>
T* ChunkAllocator<T>::reserve_if_free(vk::DeviceMemory memory, FreeListEntry freeListEntry, vk::DeviceSize size)
{
    std::scoped_lock lock(allocationMutex);

    auto chunk = std::find_if(chunks.begin(), chunks.end(),
                              [&](const auto& chunk_entry) { return chunk_entry.memory == memory; });

    SAIGA_ASSERT(chunk != chunks.end(), "Wrong allocator");

    if (chunk->freeList.empty())
    {
        return nullptr;
    }

    auto found =
        std::lower_bound(chunk->freeList.begin(), chunk->freeList.end(), freeListEntry,
                         [](const auto& free_entry, const auto& value) { return free_entry.offset < value.offset; });

    if (found == chunk->freeList.end())
    {
        return nullptr;
    }

    if (found->offset != freeListEntry.offset || found->size != freeListEntry.size)
    {
        return nullptr;
    }

    return allocate_in_free_space(size, chunk, found);
}


template <typename T>
void ChunkAllocator<T>::swap(T* target, T* source)
{
    std::scoped_lock lock(allocationMutex);

    SafeAccessor acc(*target, *source);

    ChunkIterator<T> target_chunk, source_chunk;
    AllocationIterator<T> target_alloc, source_alloc;

    std::tie(target_chunk, target_alloc) = find_allocation(target);
    std::tie(source_chunk, source_alloc) = find_allocation(source);

    std::swap(source->offset, target->offset);
    std::swap(source->memory, target->memory);


    std::swap(source->data, target->data);

    std::iter_swap(source_alloc, target_alloc);

    source->modified();
}

template <typename T>
template <typename FreeEntry>
void ChunkAllocator<T>::add_to_free_list(const ChunkIterator<T>& chunk, const FreeEntry& location) const
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
std::pair<ChunkIterator<T>, AllocationIterator<T>> ChunkAllocator<T>::find_allocation(T* location)
{
    auto fChunk = std::find_if(chunks.begin(), chunks.end(),
                               [&](Chunk<T> const& alloc) { return alloc.memory == location->memory; });

    SAIGA_ASSERT(fChunk != chunks.end(), "Allocation was not done with this allocator!");

    auto& chunkAllocs = fChunk->allocations;
    auto fLoc =
        std::lower_bound(chunkAllocs.begin(), chunkAllocs.end(), location,
                         [](const auto& element, const auto& value) { return element->offset < value->offset; });

    return std::make_pair(fChunk, fLoc);
}

template <typename T>
void ChunkAllocator<T>::showDetailStats(bool expand)
{
    using BarColor = ImGui::ColoredBar::BarColor;
    static const BarColor alloc_color_static{{1.00f, 0.447f, 0.133f, 1.0f}, {0.133f, 0.40f, 0.40f, 1.0f}};
    static const BarColor alloc_color_dynamic{{1.00f, 0.812f, 0.133f, 1.0f}, {0.133f, 0.40f, 0.40f, 1.0f}};

    static std::vector<ImGui::ColoredBar> allocation_bars;


    ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_None;
    if (expand)
    {
        node_flags = ImGuiTreeNodeFlags_DefaultOpen;
    }
    if (ImGui::CollapsingHeader(gui_identifier.c_str(), node_flags))
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
            ss << "Mem " << std::hex << chunk.memory << " Buffer " << chunk.buffer;

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
                               true);
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
MemoryStats ChunkAllocator<T>::collectMemoryStats()
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

template class ChunkAllocator<BufferMemoryLocation>;
template class ChunkAllocator<ImageMemoryLocation>;

}  // namespace Saiga::Vulkan::Memory
