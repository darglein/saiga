//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once
#include <tuple>
#include "ChunkAllocation.h"
#include "saiga/export.h"
namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
struct SAIGA_GLOBAL FitStrategy
{
    virtual std::pair<ChunkIterator, LocationIterator> findRange(std::vector<ChunkAllocation>& _allocations,
                                                                 vk::DeviceSize size) = 0;
};

struct SAIGA_GLOBAL FirstFitStrategy : public FitStrategy
{
    std::pair<ChunkIterator, LocationIterator> findRange(std::vector<ChunkAllocation>& _allocations,
                                                         vk::DeviceSize size) override;
};

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
