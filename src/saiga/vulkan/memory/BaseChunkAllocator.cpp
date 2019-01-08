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



void BaseChunkAllocator::showDetailStats()
{
    using BarColor = ImGui::ColoredBar::BarColor;
    static const std::array<BarColor, 2> colors{BarColor{{0.0f, 0.2f, 0.2f, 1.0f}, {0.133f, 0.40f, 0.40f, 1.0f}},
                                                BarColor{{0.333f, 0.0f, 0.0f, 1.0f}, {0.667f, 0.224f, 0.224f, 1.0f}}};

    static std::vector<ImGui::ColoredBar> allocation_bars;

    if (ImGui::CollapsingHeader(gui_identifier.c_str()))
    {
        ImGui::Indent();

        headerInfo();

        allocation_bars.resize(
            m_chunkAllocations.size(),
            ImGui::ColoredBar({0, 60}, {{0.1f, 0.1f, 0.1f, 1.0f}, {0.4f, 0.4f, 0.4f, 1.0f}}, true, 4));

        int numAllocs           = 0;
        uint64_t usedSpace      = 0;
        uint64_t innerFreeSpace = 0;
        uint64_t totalFreeSpace = 0;
        for (int i = 0; i < allocation_bars.size(); ++i)
        {
            ImGui::Text("Chunk %d", i + 1);
            ImGui::Indent();
            auto bar   = allocation_bars[i];
            auto chunk = m_chunkAllocations[i];
            bar.renderBackground();
            int j = 0;
            std::list<MemoryLocation>::const_iterator allocIter, freeIter;
            for (allocIter = chunk.allocations.cbegin(), j = 0; allocIter != chunk.allocations.cend(); ++allocIter, ++j)
            {
                bar.renderArea(static_cast<float>(allocIter->offset) / m_chunkSize,
                               static_cast<float>(allocIter->offset + allocIter->size) / m_chunkSize, colors[j % 2]);
                usedSpace += allocIter->size;
            }
            numAllocs += j;
            auto freeEnd = --chunk.freeList.cend();
            for (freeIter = chunk.freeList.cbegin(); freeIter != freeEnd; freeIter++)
            {
                innerFreeSpace += freeIter->size;
                totalFreeSpace += freeIter->size;
            }

            totalFreeSpace += chunk.freeList.back().size;

            ImGui::Unindent();
        }
        ImGui::LabelText("Number of allocations", "%d", numAllocs);
        auto totalSpace = m_chunkSize * m_chunkAllocations.size();


        ImGui::LabelText("Usage", "%s / %s (%.2f%%)", sizeToString(usedSpace).c_str(), sizeToString(totalSpace).c_str(),
                         100 * static_cast<float>(usedSpace) / totalSpace);
        ImGui::LabelText("Free Space (total / fragmented)", "%s / %s", sizeToString(totalFreeSpace).c_str(),
                         sizeToString(innerFreeSpace).c_str());
        ImGui::Unindent();
    }
}

MemoryStats BaseChunkAllocator::collectMemoryStats()
{
    int numAllocs                = 0;
    uint64_t usedSpace           = 0;
    uint64_t fragmentedFreeSpace = 0;
    uint64_t totalFreeSpace      = 0;
    for (int i = 0; i < m_chunkAllocations.size(); ++i)
    {
        auto chunk = m_chunkAllocations[i];
        int j      = 0;
        std::list<MemoryLocation>::const_iterator allocIter, freeIter;
        for (allocIter = chunk.allocations.cbegin(), j = 0; allocIter != chunk.allocations.cend(); ++allocIter, ++j)
        {
            usedSpace += allocIter->size;
        }
        numAllocs += j;
        auto freeEnd = --chunk.freeList.cend();
        for (freeIter = chunk.freeList.cbegin(); freeIter != freeEnd; freeIter++)
        {
            fragmentedFreeSpace += freeIter->size;
            totalFreeSpace += freeIter->size;
        }

        totalFreeSpace += chunk.freeList.back().size;
    }
    auto totalSpace = m_chunkSize * m_chunkAllocations.size();

    return MemoryStats{totalSpace, usedSpace, fragmentedFreeSpace};
};
}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga