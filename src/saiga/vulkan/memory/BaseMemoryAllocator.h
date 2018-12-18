//
// Created by Peter Eichinger on 15.10.18.
//

#pragma once

#include "MemoryLocation.h"

#include <saiga/util/assert.h>
namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
struct MemoryStats
{
    vk::DeviceSize allocated;
    vk::DeviceSize used;
    vk::DeviceSize fragmented;

    MemoryStats() : allocated(0), used(0), fragmented(0) {}

    MemoryStats(vk::DeviceSize _allocated, vk::DeviceSize _used, vk::DeviceSize _fragmented)
        : allocated(_allocated), used(_used), fragmented(_fragmented)
    {
    }

    MemoryStats& operator+=(const MemoryStats& rhs)
    {
        allocated += rhs.allocated;
        used += rhs.used;
        fragmented += rhs.fragmented;
        return *this;
    }
};

struct SAIGA_GLOBAL BaseMemoryAllocator
{
    explicit BaseMemoryAllocator(vk::DeviceSize _maxAllocationSize = VK_WHOLE_SIZE)
        : maxAllocationSize(_maxAllocationSize)
    {
    }

    BaseMemoryAllocator(const BaseMemoryAllocator& other)     = delete;
    BaseMemoryAllocator(BaseMemoryAllocator&& other) noexcept = default;

    BaseMemoryAllocator& operator=(const BaseMemoryAllocator& other) = delete;
    BaseMemoryAllocator& operator=(BaseMemoryAllocator&& other) = default;

    virtual ~BaseMemoryAllocator() = default;

    virtual MemoryLocation allocate(vk::DeviceSize size) = 0;
    virtual void deallocate(MemoryLocation& location)    = 0;
    vk::DeviceSize maxAllocationSize                     = VK_WHOLE_SIZE;

    virtual void destroy() {}

    virtual MemoryStats collectMemoryStats() { return MemoryStats(); };

    virtual void showDetailStats(){};
};

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga