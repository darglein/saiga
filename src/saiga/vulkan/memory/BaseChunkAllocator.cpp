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
    MemoryLocation* val;
    {
        defragger->stop();
        std::scoped_lock alloc_lock(allocationMutex);
        ChunkIterator chunkAlloc;
        FreeIterator freeSpace;
        std::tie(chunkAlloc, freeSpace) =
            m_strategy->findRange(m_chunkAllocations.begin(), m_chunkAllocations.end(), size);

        if (chunkAlloc == m_chunkAllocations.end())
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


        auto targetLocation = std::make_unique<MemoryLocation>(chunkAlloc->buffer, chunkAlloc->chunk->memory,
                                                               memoryStart, size, chunkAlloc->mappedPointer);
        auto memoryEnd      = memoryStart + size;

        auto insertionPoint =
            std::lower_bound(chunkAlloc->allocations.begin(), chunkAlloc->allocations.end(), memoryEnd,
                             [](const auto& element, vk::DeviceSize value) { return element->offset < value; });

        val = chunkAlloc->allocations.insert(insertionPoint, std::move(targetLocation))->get();
        chunkAlloc->allocated += size;
        defragger->invalidate(chunkAlloc->chunk->memory);
        defragger->start();
    }
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
    {
        std::scoped_lock alloc_lock(allocationMutex);
        defragger->stop();

        auto fChunk =
            std::find_if(m_chunkAllocations.begin(), m_chunkAllocations.end(),
                         [&](ChunkAllocation const& alloc) { return alloc.chunk->memory == location->memory; });

        SAIGA_ASSERT(fChunk != m_chunkAllocations.end(), "Allocation was not done with this allocator!");
        auto& chunkAllocs = fChunk->allocations;
        auto& chunkFree   = fChunk->freeList;
        auto fLoc =
            std::lower_bound(chunkAllocs.begin(), chunkAllocs.end(), location,
                             [](const auto& element, const auto& value) { return element->offset < value->offset; });
        SAIGA_ASSERT(fLoc != chunkAllocs.end(), "Allocation is not part of the chunk");
        LOG(INFO) << "Deallocating " << location->size << " bytes in chunk/offset ["
                  << distance(m_chunkAllocations.begin(), fChunk) << "/" << (*fLoc)->offset << "]";

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
        defragger->invalidate(fChunk->chunk->memory);

        if (m_chunkAllocations.size() >= 2)
        {
            // Free memory if second to last and last chunk are empty
            auto last           = m_chunkAllocations.end() - 1;
            auto second_to_last = last - 1;
            if (last->allocations.empty() && second_to_last->allocations.empty())
            {
                m_device.destroy(last->buffer);
                m_chunkAllocator->deallocate(last->chunk);
                m_chunkAllocations.erase(last);
            }
        }

        defragger->start();
    }
}
void BaseChunkAllocator::destroy()
{
    for (auto& alloc : m_chunkAllocations)
    {
        m_device.destroy(alloc.buffer);
    }
}

}  // namespace Saiga::Vulkan::Memory
