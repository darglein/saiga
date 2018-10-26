//
// Created by Peter Eichinger on 25.10.18.
//

#pragma once

#include <memory>

#include <vulkan/vulkan.hpp>
#include "saiga/vulkan/memory/FindMemoryType.h"
#include "saiga/util/easylogging++.h"
#include "saiga/vulkan/Base.h"


namespace Saiga{
namespace Vulkan{
namespace Memory{

/**
 * Allocator that uses host visible and host coherent vulkan memory.
 * @tparam T Type of object to allocate.
 */
template <typename T>
struct VulkanStlAllocator {
    typedef size_t            size_type;
    typedef ptrdiff_t         difference_type;
    typedef T value_type;
    typedef T* pointer;
    typedef T const * const_pointer;
    typedef T& reference;
    typedef T const & const_reference;

private:

    struct AllocationHeader{
        MemoryLocation memoryLocation;
    };

    vk::Device m_device;
    vk::PhysicalDevice m_physicalDevice;
    vk::BufferUsageFlags m_usageFlags;

    MemoryAllocatorBase* allocator;
public:
    VulkanStlAllocator(VulkanBase& base, const vk::BufferUsageFlags &_usageFlags) :
        m_device(base.device), m_physicalDevice(base.physicalDevice), m_usageFlags(_usageFlags) {
        allocator = &base.memory.getAllocator(_usageFlags, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        SAIGA_ASSERT(allocator->mapped, "Only mapped allocators are allowed to be used");
    }

    pointer allocate(size_type n, const_pointer hint = 0) {
        const auto size = n * sizeof(T) + sizeof(AllocationHeader);
//        vk::BufferCreateInfo bci;
//        bci.size = size;
//        bci.sharingMode = vk::SharingMode::eExclusive;
//        bci.usage = m_usageFlags;
//        vk::Buffer new_buffer = m_device.createBuffer(bci);
//        vk::MemoryRequirements mem_reqs = m_device.getBufferMemoryRequirements(new_buffer);
//
//        vk::MemoryAllocateInfo mai;
//        mai.memoryTypeIndex = findMemoryType(m_physicalDevice,mem_reqs.memoryTypeBits,
//                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
//        mai.allocationSize = mem_reqs.size;
//
//        vk::DeviceMemory new_mem = m_device.allocateMemory(mai);

        auto allocation = allocator->allocate(size);

        SAIGA_ASSERT(allocation.mappedPointer, "Allocation is not mapped");

//        auto alloc =  m_device.mapMemory(new_mem,0,size);

        auto headerPtr = reinterpret_cast<AllocationHeader*>(allocation.mappedPointer);
        headerPtr->memoryLocation = allocation;

        LOG(INFO) << "Allocated " << allocation.memory << "@" << allocation.mappedPointer << " MemStart: " << headerPtr + 1 << std::endl;
        return reinterpret_cast<pointer>(headerPtr + 1);
    }

    void deallocate(pointer p, size_type n) {
        auto headerPtr = reinterpret_cast<AllocationHeader*>(p);
        auto header = *(headerPtr-1);

        LOG(INFO)<< "Deallocating " << header.memoryLocation.memory << "@" << header.memoryLocation.mappedPointer << std::endl;
        allocator->deallocate(header.memoryLocation);
//        m_device.destroy(header.buffer);
//        m_device.free(header.memory);
    }

    void construct(pointer p, const_reference t) { new ((void*) p) T(t); }

    void destroy(pointer p){ ((T*)p)->~T(); }
};


}
}
}