//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once
#include "saiga/export.h"

#include "ChunkAllocation.h"

#include <tuple>
namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
struct SAIGA_LOCAL FitStrategy
{
    virtual ~FitStrategy() = default;

    virtual std::pair<ChunkIterator, LocationIterator> findRange(ChunkIterator begin, ChunkIterator end,
                                                                 vk::DeviceSize size) = 0;

};

struct SAIGA_LOCAL FirstFitStrategy : public FitStrategy
{
    std::pair<ChunkIterator, LocationIterator> findRange(ChunkIterator begin, ChunkIterator end,
                                                         vk::DeviceSize size) override;
};

struct SAIGA_LOCAL BestFitStrategy : public FitStrategy
{
    std::pair<ChunkIterator, LocationIterator> findRange(ChunkIterator begin, ChunkIterator end,
                                                         vk::DeviceSize size) override;
};

struct SAIGA_LOCAL WorstFitStrategy : public FitStrategy
{
    std::pair<ChunkIterator, LocationIterator> findRange(ChunkIterator begin, ChunkIterator end,
                                                         vk::DeviceSize size) override;
};

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
