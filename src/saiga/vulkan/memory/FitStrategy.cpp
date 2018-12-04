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
    auto foundChunk = std::find_if(_allocations.begin(), _allocations.end(),
                                   [&](ChunkAllocation& alloc) { return alloc.maxFreeRange->size > size; });

    if (foundChunk == _allocations.end())
    {
        return std::make_pair(_allocations.end(), LocationIterator());
    }

    auto& chunk = *foundChunk;

    auto foundRange = std::find_if(chunk.freeList.begin(), chunk.freeList.end(),
                                   [&](MemoryLocation& loc) { return loc.size >= size; });

    SAIGA_ASSERT(foundRange != chunk.freeList.end(), "free size is invalid.");

    return std::make_pair(foundChunk, foundRange);
}

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga