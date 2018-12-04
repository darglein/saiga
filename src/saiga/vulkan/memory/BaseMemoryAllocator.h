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
struct SAIGA_GLOBAL BaseMemoryAllocator
{
    explicit BaseMemoryAllocator(bool _mapped) : mapped(_mapped) {}

    BaseMemoryAllocator(const BaseMemoryAllocator& other) = delete;
    BaseMemoryAllocator(BaseMemoryAllocator&& other) noexcept : mapped(other.mapped) {}

    BaseMemoryAllocator& operator=(const BaseMemoryAllocator& other) = delete;
    BaseMemoryAllocator& operator=(BaseMemoryAllocator&& other) = default;

    virtual ~BaseMemoryAllocator() = default;

    virtual MemoryLocation allocate(vk::DeviceSize size) = 0;
    virtual void deallocate(MemoryLocation& location)    = 0;
    bool mapped                                          = false;

    virtual void destroy() {}
};

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga