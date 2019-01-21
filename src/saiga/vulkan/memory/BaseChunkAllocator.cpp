//
// Created by Peter Eichinger on 30.10.18.
//

#include "BaseChunkAllocator.h"

#include "saiga/imgui/imgui.h"
#include "saiga/util/easylogging++.h"
#include "saiga/util/tostring.h"

#include "BufferChunkAllocator.h"
#include "ChunkCreator.h"

namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
MemoryLocation BaseChunkAllocator::allocate(vk::DeviceSize size)
{
    allocationMutex.lock();
    ChunkIterator chunkAlloc;
    LocationIterator freeSpace;
    tie(chunkAlloc, freeSpace) = m_strategy->findRange(m_chunkAllocations, size);

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

    MemoryLocation targetLocation = createMemoryLocation(chunkAlloc, memoryStart, size);
    auto memoryEnd                = memoryStart + size;
    auto insertionPoint           = find_if(chunkAlloc->allocations.begin(), chunkAlloc->allocations.end(),
                                  [=](MemoryLocation& loc) { return loc.offset > memoryEnd; });


    auto val = *chunkAlloc->allocations.emplace(insertionPoint, targetLocation);
    allocationMutex.unlock();

    return val;
}

void BaseChunkAllocator::findNewMax(ChunkIterator& chunkAlloc) const
{
    auto& freeList = chunkAlloc->freeList;
    chunkAlloc->maxFreeRange =
        max_element(freeList.begin(), freeList.end(),
                    [](MemoryLocation& first, MemoryLocation& second) { return first.size < second.size; });
}

MemoryLocation BaseChunkAllocator::createMemoryLocation(ChunkIterator iter, vk::DeviceSize offset, vk::DeviceSize size)
{
    return MemoryLocation{iter->buffer, iter->chunk->memory, offset, size, iter->mappedPointer};
}

void BaseChunkAllocator::deallocate(MemoryLocation& location)
{
    std::scoped_lock alloc_lock(allocationMutex);
    auto fChunk = find_if(m_chunkAllocations.begin(), m_chunkAllocations.end(),
                          [&](ChunkAllocation const& alloc) { return alloc.chunk->memory == location.memory; });

    SAIGA_ASSERT(fChunk != m_chunkAllocations.end(), "Allocation was not done with this allocator!");
    auto& chunkAllocs = fChunk->allocations;
    auto& chunkFree   = fChunk->freeList;
    auto fLoc         = find(chunkAllocs.begin(), chunkAllocs.end(), location);
    SAIGA_ASSERT(fLoc != chunkAllocs.end(), "Allocation is not part of the chunk");
    LOG(INFO) << "Deallocating " << location.size << " bytes in chunk/offset ["
              << distance(m_chunkAllocations.begin(), fChunk) << "/" << fLoc->offset << "]";

    LocationIterator freePrev, freeNext, freeInsert;
    bool foundInsert = false;
    freePrev = freeNext = chunkFree.end();
    freeInsert          = chunkFree.end();
    for (auto freeIt = chunkFree.begin(); freeIt != chunkFree.end(); ++freeIt)
    {
        if (freeIt->offset + freeIt->size == location.offset)
        {
            freePrev = freeIt;
        }
        if (freeIt->offset == location.offset + location.size)
        {
            freeNext = freeIt;
            break;
        }
        if ((freeIt->offset + freeIt->size) < location.offset)
        {
            freeInsert  = freeIt;
            foundInsert = true;
        }
    }



    if (freePrev != chunkFree.end() && freeNext != chunkFree.end())
    {
        // Free space before and after newly freed space -> merge
        freePrev->size += location.size + freeNext->size;
        chunkFree.erase(freeNext);
    }
    else if (freePrev != chunkFree.end())
    {
        // Free only before -> increase size
        freePrev->size += location.size;
    }
    else if (freeNext != chunkFree.end())
    {
        // Free only after newly freed -> move and increase size
        freeNext->offset = location.offset;
        freeNext->size += location.size;
    }
    else
    {
        if (foundInsert)
        {
            chunkFree.insert(std::next(freeInsert), location);
        }
        else
        {
            chunkFree.push_front(location);
        }
    }

    findNewMax(fChunk);

    chunkAllocs.erase(fLoc);
}
void BaseChunkAllocator::destroy()
{
    for (auto& alloc : m_chunkAllocations)
    {
        m_device.destroy(alloc.buffer);
    }
}

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
