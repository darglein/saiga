//
// Created by Peter Eichinger on 30.10.18.
//

#include "BaseChunkAllocator.h"

#include "saiga/imgui/imgui.h"
#include "saiga/util/easylogging++.h"
#include "saiga/util/tostring.h"

#include "BufferChunkAllocator.h"
#include "ChunkCreator.h"
namespace Saiga::Vulkan::Memory
{
MemoryLocation* BaseChunkAllocator::allocate(vk::DeviceSize size)
{
    std::scoped_lock alloc_lock(allocationMutex);
    ChunkIterator chunkAlloc;
    FreeIterator freeSpace;
    std::tie(chunkAlloc, freeSpace) = strategy->findRange(chunks.begin(), chunks.end(), size);

    MemoryLocation* val = allocate_in_free_space(size, chunkAlloc, freeSpace);

    return val;
}

MemoryLocation* BaseChunkAllocator::allocate_in_free_space(vk::DeviceSize size, ChunkIterator& chunkAlloc,
                                                           FreeIterator& freeSpace)
{
    MemoryLocation* val;
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


    // MemoryLocation{iter->buffer, iter->chunk->memory, offset, size, iter->mappedPointer};


    auto targetLocation = std::__1::make_unique<MemoryLocation>(chunkAlloc->buffer, chunkAlloc->chunk->memory,
                                                                memoryStart, size, chunkAlloc->mappedPointer);
    auto memoryEnd      = memoryStart + size;

    auto insertionPoint =
        lower_bound(chunkAlloc->allocations.begin(), chunkAlloc->allocations.end(), memoryEnd,
                    [](const auto& element, vk::DeviceSize value) { return element->offset < value; });

    val = chunkAlloc->allocations.insert(insertionPoint, move(targetLocation))->get();
    chunkAlloc->allocated += size;
    return val;
}

void BaseChunkAllocator::findNewMax(ChunkIterator& chunkAlloc) const
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


void BaseChunkAllocator::deallocate(MemoryLocation* location)
{
    std::scoped_lock alloc_lock(allocationMutex);

    ChunkIterator fChunk;
    AllocationIterator fLoc;

    std::tie(fChunk, fLoc) = find_allocation(location);

    auto& chunkAllocs = fChunk->allocations;

    SAIGA_ASSERT(fLoc != chunkAllocs.end(), "Allocation is not part of the chunk");
    auto& chunkFree = fChunk->freeList;
    LOG(INFO) << "Deallocating " << location->size << " bytes in chunk/offset [" << distance(chunks.begin(), fChunk)
              << "/" << (*fLoc)->offset << "]";

    FreeIterator freePrev, freeNext, freeInsert;
    bool foundInsert = false;
    freePrev = freeNext = chunkFree.end();
    freeInsert          = chunkFree.end();
    for (auto freeIt = chunkFree.begin(); freeIt != chunkFree.end(); ++freeIt)
    {
        if (freeIt->offset + freeIt->size == location->offset)
        {
            freePrev = freeIt;
        }
        if (freeIt->offset == location->offset + location->size)
        {
            freeNext = freeIt;
            break;
        }
        if ((freeIt->offset + freeIt->size) < location->offset)
        {
            freeInsert  = freeIt;
            foundInsert = true;
        }
    }

    if (freePrev != chunkFree.end() && freeNext != chunkFree.end())
    {
        // Free space before and after newly freed space -> merge
        freePrev->size += location->size + freeNext->size;
        chunkFree.erase(freeNext);
    }
    else if (freePrev != chunkFree.end())
    {
        // Free only before -> increase size
        freePrev->size += location->size;
    }
    else if (freeNext != chunkFree.end())
    {
        // Free only after newly freed -> move and increase size
        freeNext->offset = location->offset;
        freeNext->size += location->size;
    }
    else
    {
        FreeIterator insertionPoint;
        if (foundInsert)
        {
            insertionPoint = std::next(freeInsert);
        }
        else
        {
            insertionPoint = chunkFree.begin();
        }

        chunkFree.insert(insertionPoint, FreeListEntry{location->offset, location->size});
    }

    findNewMax(fChunk);

    chunkAllocs.erase(fLoc);

    fChunk->allocated -= location->size;
    if (chunks.size() >= 2)
    {
        // Free memory if second to last and last chunk are empty
        auto last           = chunks.end() - 1;
        auto second_to_last = last - 1;
        if (last->allocations.empty() && second_to_last->allocations.empty())
        {
            m_device.destroy(last->buffer);
            m_chunkAllocator->deallocate(last->chunk);
            chunks.erase(last);
        }
    }
}
void BaseChunkAllocator::destroy()
{
    for (auto& alloc : chunks)
    {
        m_device.destroy(alloc.buffer);
    }
}

bool BaseChunkAllocator::memory_is_free(vk::DeviceMemory memory, FreeListEntry free_mem)
{
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

    // if (found->offset > free_mem.offset)
    //{
    //    return false;
    //}
    //
    // if (found->offset + found->size < free_mem.offset + free_mem.size)
    //{
    //    return false;
    //}

    return found->offset == free_mem.offset && found->size == free_mem.size;
}

MemoryLocation* BaseChunkAllocator::reserve_space(vk::DeviceMemory memory, FreeListEntry freeListEntry,
                                                  vk::DeviceSize size)
{
    auto chunk = std::find_if(chunks.begin(), chunks.end(),
                              [&](const auto& chunk_entry) { return chunk_entry.chunk->memory == memory; });

    SAIGA_ASSERT(chunk != chunks.end(), "Wrong allocator");

    auto free = std::find(chunk->freeList.begin(), chunk->freeList.end(), freeListEntry);

    SAIGA_ASSERT(free != chunk->freeList.end(), "Free space not found");

    return allocate_in_free_space(size, chunk, free);
}

void BaseChunkAllocator::move_allocation(MemoryLocation* target, MemoryLocation* source)
{
    const auto size = source->size;

    MemoryLocation source_copy = *source;

    ChunkIterator target_chunk, source_chunk;
    AllocationIterator target_alloc, source_alloc;

    std::tie(target_chunk, target_alloc) = find_allocation(target);
    std::tie(source_chunk, source_alloc) = find_allocation(source);

    *source = *target;  // copy values from target to source;

    // target_chunk->allocated -= size;
    source_chunk->allocated -= size;

    std::move(source_alloc, std::next(source_alloc), target_alloc);

    //**target_alloc = source_copy;
    source_chunk->allocations.erase(source_alloc);

    auto found =
        std::lower_bound(source_chunk->freeList.begin(), source_chunk->freeList.end(), source_copy,
                         [](const auto& free_entry, const auto& value) { return free_entry.offset < value.offset; });

    source_chunk->freeList.insert(found, FreeListEntry{source_copy.offset, source_copy.size});

    findNewMax(source_chunk);
}

std::pair<ChunkIterator, AllocationIterator> BaseChunkAllocator::find_allocation(MemoryLocation* location)
{
    auto fChunk = std::find_if(chunks.begin(), chunks.end(),
                               [&](ChunkAllocation const& alloc) { return alloc.chunk->memory == location->memory; });

    SAIGA_ASSERT(fChunk != chunks.end(), "Allocation was not done with this allocator!");

    auto& chunkAllocs = fChunk->allocations;
    auto fLoc =
        std::lower_bound(chunkAllocs.begin(), chunkAllocs.end(), location,
                         [](const auto& element, const auto& value) { return element->offset < value->offset; });

    return std::make_pair(fChunk, fLoc);
}

}  // namespace Saiga::Vulkan::Memory
