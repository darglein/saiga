//
// Created by Peter Eichinger on 15.10.18.
//

#pragma once

#include "MemoryLocation.h"

#include <saiga/core/util/assert.h>

namespace Saiga::Vulkan::Memory
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

}  // namespace Saiga::Vulkan::Memory
