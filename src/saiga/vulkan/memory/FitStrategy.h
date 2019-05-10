//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once
#include "saiga/export.h"

#include "Chunk.h"

#include <tuple>
namespace Saiga::Vulkan::Memory
{
template <typename T>
struct SAIGA_LOCAL FitStrategy
{
    virtual ~FitStrategy() = default;

    virtual std::pair<ChunkIterator<T>, FreeIterator<T>> findRange(ChunkIterator<T> begin, ChunkIterator<T> end,
                                                                   vk::DeviceSize size) = 0;
};

template <typename T>
struct SAIGA_LOCAL FirstFitStrategy final : public FitStrategy<T>
{
    std::pair<ChunkIterator<T>, FreeIterator<T>> findRange(ChunkIterator<T> begin, ChunkIterator<T> end,
                                                           vk::DeviceSize size) override;
};

template <typename T>
struct SAIGA_LOCAL BestFitStrategy final : public FitStrategy<T>
{
    std::pair<ChunkIterator<T>, FreeIterator<T>> findRange(ChunkIterator<T> begin, ChunkIterator<T> end,
                                                           vk::DeviceSize size) override;
};

template <typename T>
struct SAIGA_LOCAL WorstFitStrategy final : public FitStrategy<T>
{
    std::pair<ChunkIterator<T>, FreeIterator<T>> findRange(ChunkIterator<T> begin, ChunkIterator<T> end,
                                                           vk::DeviceSize size) override;
};


template <typename T>
std::pair<ChunkIterator<T>, FreeIterator<T>> FirstFitStrategy<T>::findRange(ChunkIterator<T> begin,
                                                                            ChunkIterator<T> end, vk::DeviceSize size)
{
    auto foundChunk = std::find_if(begin, end, [&](Chunk<T>& alloc) {
        return alloc.maxFreeRange.value_or(FreeListEntry{0, 0}).size >= size;
    });

    if (foundChunk == end)
    {
        return std::make_pair(end, FreeIterator<T>());
    }

    auto& chunk = *foundChunk;

    auto foundRange = std::find_if(chunk.freeList.begin(), chunk.freeList.end(),
                                   [&](const FreeListEntry& loc) { return loc.size >= size; });

    SAIGA_ASSERT(foundRange != chunk.freeList.end(), "free size is invalid.");

    return std::make_pair(foundChunk, foundRange);
}


template <typename T, typename CompareFunc>
std::pair<ChunkIterator<T>, FreeIterator<T>> findFitPairForBestWorst(ChunkIterator<T> begin, ChunkIterator<T> end,
                                                                     vk::DeviceSize size)
{
    bool found = false;
    ChunkIterator<T> foundChunk;
    FreeIterator<T> foundFreeSpace;

    auto chunkIt = begin;

    while (chunkIt != end)
    {
        for (auto freeIt = chunkIt->freeList.begin(); freeIt != chunkIt->freeList.end(); ++freeIt)
        {
            auto free_size = freeIt->size;
            if (free_size < size)
            {
                continue;
            }
            if (free_size == size)
            {
                return std::make_pair(chunkIt, freeIt);
            }
            if (!found || CompareFunc()(free_size, foundFreeSpace->size))
            {
                found          = true;
                foundFreeSpace = freeIt;
                foundChunk     = chunkIt;
            }
        }
        chunkIt++;
    }

    if (!found)
    {
        return std::make_pair(end, FreeIterator<T>());
    }
    return std::make_pair(foundChunk, foundFreeSpace);
}



template <typename T>
std::pair<ChunkIterator<T>, FreeIterator<T>> BestFitStrategy<T>::findRange(ChunkIterator<T> begin, ChunkIterator<T> end,
                                                                           vk::DeviceSize size)
{
    return findFitPairForBestWorst<T, std::less<vk::DeviceSize>>(begin, end, size);
}

template <typename T>
std::pair<ChunkIterator<T>, FreeIterator<T>> WorstFitStrategy<T>::findRange(ChunkIterator<T> begin,
                                                                            ChunkIterator<T> end, vk::DeviceSize size)
{
    return findFitPairForBestWorst<T, std::greater<vk::DeviceSize>>(begin, end, size);
}

}  // namespace Saiga::Vulkan::Memory
