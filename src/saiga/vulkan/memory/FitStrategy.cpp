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
std::pair<ChunkIterator, LocationIterator> FirstFitStrategy::findRange(std::vector<ChunkAllocation>& _allocations,
                                                                       vk::DeviceSize size)
{
    auto foundChunk = std::find_if(_allocations.begin(), _allocations.end(), [&](ChunkAllocation& alloc) {
        return alloc.maxFreeRange != alloc.freeList.end() && (alloc.maxFreeRange->size >= size);
    });

    if (foundChunk == _allocations.end())
    {
        return std::make_pair(ChunkIterator(), LocationIterator());
    }

    auto& chunk = *foundChunk;

    auto foundRange = std::find_if(chunk.freeList.begin(), chunk.freeList.end(),
                                   [&](MemoryLocation& loc) { return loc.size >= size; });

    SAIGA_ASSERT(foundRange != chunk.freeList.end(), "free size is invalid.");

    return std::make_pair(foundChunk, foundRange);
}

template <typename CompareFunc>
std::pair<ChunkIterator, LocationIterator> findFitPairForBestWorst(std::vector<ChunkAllocation>& _allocations,
                                                                   vk::DeviceSize size)
{
    bool found = false;
    ChunkIterator foundChunk;
    LocationIterator foundFreeSpace;

    for (auto chunkIt = _allocations.begin(); chunkIt != _allocations.end(); ++chunkIt)
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
    }

    return make_pair(foundChunk, foundFreeSpace);
}

std::pair<ChunkIterator, LocationIterator> BestFitStrategy::findRange(std::vector<ChunkAllocation>& _allocations,
                                                                      vk::DeviceSize size)
{
    return findFitPairForBestWorst<std::less<vk::DeviceSize>>(_allocations, size);
}

std::pair<ChunkIterator, LocationIterator> WorstFitStrategy::findRange(std::vector<ChunkAllocation>& _allocations,
                                                                       vk::DeviceSize size)
{
    return findFitPairForBestWorst<std::greater<vk::DeviceSize>>(_allocations, size);
}

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
