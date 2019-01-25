//
// Created by Peter Eichinger on 30.10.18.
//

#include "FitStrategy.h"

#include <tuple>
#include <utility>
namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
std::pair<ChunkIterator, LocationIterator> FirstFitStrategy::findRange(ChunkIterator begin, ChunkIterator end,
                                                                       vk::DeviceSize size)
{
    auto foundChunk = std::find_if(
        begin, end, [&](ChunkAllocation& alloc) { return alloc.maxFreeRange && (alloc.maxFreeRange.size >= size); });

    if (foundChunk == end)
    {
        return std::make_pair(end, LocationIterator());
    }

    auto& chunk = *foundChunk;

    auto foundRange = std::find_if(chunk.freeList.begin(), chunk.freeList.end(),
                                   [&](MemoryLocation& loc) { return loc.size >= size; });

    SAIGA_ASSERT(foundRange != chunk.freeList.end(), "free size is invalid.");

    return std::make_pair(foundChunk, foundRange);
}

template <typename CompareFunc>
std::pair<ChunkIterator, LocationIterator> findFitPairForBestWorst(ChunkIterator begin, ChunkIterator end,
                                                                   vk::DeviceSize size)
{
    bool found = false;
    ChunkIterator foundChunk;
    LocationIterator foundFreeSpace;

    auto chunkIt = begin;

    while (chunkIt != end)
    {
        for (auto freeIt = chunkIt->freeList.begin(); freeIt != chunkIt->freeList.end(); ++freeIt)
        {
            if (!found)
            {
                found          = true;
                foundChunk     = chunkIt;
                foundFreeSpace = freeIt;
                continue;
            }

            auto free_size = freeIt->size;
            if (free_size < size)
            {
                continue;
            }
            if (free_size == size)
            {
                return make_pair(chunkIt, freeIt);
            }
            if (CompareFunc()(free_size, foundFreeSpace->size))
            {
                foundFreeSpace = freeIt;
                foundChunk     = chunkIt;
            }
        }
        chunkIt++;
    }

    return make_pair(foundChunk, foundFreeSpace);
}

std::pair<ChunkIterator, LocationIterator> BestFitStrategy::findRange(ChunkIterator begin, ChunkIterator end,
                                                                      vk::DeviceSize size)
{
    return findFitPairForBestWorst<std::less<vk::DeviceSize>>(begin, end, size);
}

std::pair<ChunkIterator, LocationIterator> WorstFitStrategy::findRange(ChunkIterator begin, ChunkIterator end,
                                                                       vk::DeviceSize size)
{
    return findFitPairForBestWorst<std::greater<vk::DeviceSize>>(begin, end, size);
}

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
