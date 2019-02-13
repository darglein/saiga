//
// Created by Peter Eichinger on 25.10.18.
//

#pragma once

#include "saiga/core/util/easylogging++.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/memory/FindMemoryType.h"

#include <memory>
#include <vulkan/vulkan.hpp>


namespace Saiga::Vulkan::Memory
{
/**
 * Allocator that uses host visible and host coherent vulkan memory.
 * @tparam T Type of object to allocate.
 */
template <typename T>
struct VulkanStlAllocator
{
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T value_type;
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef T& reference;
    typedef T const& const_reference;

   private:
    struct AllocationHeader
    {
        BufferMemoryLocation* memoryLocation;
    };


    BufferType type;
    VulkanBase* base;

   public:
    VulkanStlAllocator(VulkanBase& _base, const vk::BufferUsageFlags& _usageFlags)
        : type({_usageFlags, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent}),
          base(&_base)
    {
    }

    pointer allocate(size_type n, const_pointer hint = 0)
    {
        const auto size = n * sizeof(T) + sizeof(AllocationHeader);

        LOG(DEBUG) << "Stl Allocating " << n * sizeof(T);

        auto allocation = base->memory.allocate(type, size);

        SAIGA_ASSERT(allocation->mappedPointer, "Allocation is not mapped");

        auto headerPtr           = reinterpret_cast<AllocationHeader*>(allocation->mappedPointer);
        headerPtr.memoryLocation = allocation;

        return reinterpret_cast<pointer>(headerPtr + 1);
    }

    void deallocate(pointer p, size_type n)
    {
        auto headerPtr = reinterpret_cast<AllocationHeader*>(p);
        auto header    = *(headerPtr - 1);

        LOG(DEBUG) << "Deallocating " << n * sizeof(T);
        base->memory.deallocateBuffer(type, header.memoryLocation);
    }

    void construct(pointer p, const_reference t) { new ((void*)p) T(t); }

    void destroy(pointer p) { ((T*)p)->~T(); }
};


}  // namespace Saiga::Vulkan::Memory