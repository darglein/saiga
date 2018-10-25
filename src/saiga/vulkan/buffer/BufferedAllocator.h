//
// Created by Peter Eichinger on 25.10.18.
//

#pragma once

#include <memory>

#include <vulkan/vulkan.hpp>
#include "saiga/vulkan/memory/FindMemoryType.h"
#include "saiga/util/easylogging++.h"

//template<typename T>
//struct Allocation {
//    T* pointer;
//    vk::DeviceMemory m_memory;
//    vk::Buffer m_buffer;
//};

struct AllocationHeader{
    vk::Buffer buffer;
    vk::DeviceMemory memory;
};

template <typename T>
struct BufferedAllocator {
    typedef size_t            size_type;
    typedef ptrdiff_t         difference_type;
    typedef T value_type;
    typedef T* pointer;
    typedef T const * const_pointer;
    typedef T& reference;
    typedef T const & const_reference;


private:
    vk::DeviceMemory m_memory;
    vk::Buffer m_buffer;
    vk::Device m_device;
    vk::PhysicalDevice m_physicalDevice;
    vk::BufferUsageFlags m_usageFlags;
public:
    BufferedAllocator(vk::Device _device,vk::PhysicalDevice _pDevice, const vk::BufferUsageFlags &_usageFlags) : m_device(_device), m_usageFlags(_usageFlags),
        m_physicalDevice(_pDevice){
    }

    pointer allocate(size_type n, const_pointer hint = 0) {
        vk::BufferCreateInfo bci;
        const auto size = n * sizeof(T) + sizeof(AllocationHeader);
        bci.size = size;
        bci.sharingMode = vk::SharingMode::eExclusive;
        bci.usage = m_usageFlags;
        vk::Buffer new_buffer = m_device.createBuffer(bci);
        vk::MemoryRequirements mem_reqs = m_device.getBufferMemoryRequirements(new_buffer);

        vk::MemoryAllocateInfo mai;
        mai.memoryTypeIndex = findMemoryType(m_physicalDevice,mem_reqs.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        mai.allocationSize = mem_reqs.size;

        vk::DeviceMemory new_mem = m_device.allocateMemory(mai);


        auto alloc =  m_device.mapMemory(new_mem,0,size);

        auto headerPtr = reinterpret_cast<AllocationHeader*>(alloc);
        headerPtr->buffer = new_buffer;
        headerPtr->memory = new_mem;
        return reinterpret_cast<pointer>(headerPtr + 1);
    }

    void deallocate(pointer p, size_type n) {
        auto headerPtr = reinterpret_cast<AllocationHeader*>(p);
        auto header = *(headerPtr-1);

        m_device.destroy(header.buffer);
        m_device.free(header.memory);
    }

    void construct(pointer p, const_reference t) { new ((void*) p) T(t); }

    void destroy(pointer p){ ((T*)p)->~T(); }
};