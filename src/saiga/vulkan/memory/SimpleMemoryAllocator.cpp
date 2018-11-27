//
// Created by Peter Eichinger on 24.11.18.
//

#include "SimpleMemoryAllocator.h"

void SimpleMemoryAllocator::deallocate(MemoryLocation &location) {
    mutex.lock();
    LOG(INFO) << "Simple Allocator: Deallocating" << location.memory << std::endl;

    auto foundAllocation = std::find(m_allocations.begin(), m_allocations.end(), location);
    if (foundAllocation == m_allocations.end()) {
        LOG(ERROR) << "Allocation was not made with this allocator";
        return;
    }
    foundAllocation->destroy(m_device);
    m_allocations.erase(foundAllocation);
    mutex.unlock();
}

void SimpleMemoryAllocator::destroy() {
    for(auto& location : m_allocations) {
        if (location.buffer != static_cast<vk::Buffer>(nullptr)) {
            location.destroy(m_device);
        }
    }
    m_allocations.clear();
}

MemoryLocation SimpleMemoryAllocator::allocate(vk::DeviceSize size) {
    LOG(INFO) << "Simple Alloc " << vk::to_string(usageFlags) << " " << std::to_string(size);
    m_bufferCreateInfo.size = size;
    auto buffer = m_device.createBuffer(m_bufferCreateInfo);

    auto memReqs = m_device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo info;
    info.allocationSize = memReqs.size;
    info.memoryTypeIndex = findMemoryType(m_physicalDevice, memReqs.memoryTypeBits, flags);
    auto memory = SafeAllocator::instance()->allocateMemory(m_device, info);

    void* mappedPtr = nullptr;
    if (mapped) {
        mappedPtr = m_device.mapMemory(memory,0, memReqs.size);
    }
    m_device.bindBufferMemory(buffer, memory,0);

    mutex.lock();
    m_allocations.emplace_back(buffer, memory, 0,size, mappedPtr);
    auto retVal = m_allocations.back();
    mutex.unlock();
    return retVal;
}

void SimpleMemoryAllocator::init(vk::Device _device, vk::PhysicalDevice _physicalDevice,
                                 const vk::MemoryPropertyFlags &_flags, const vk::BufferUsageFlags &usage, bool _mapped) {
    m_device = _device;
    m_physicalDevice = _physicalDevice;
    flags = _flags;
    mapped = _mapped;
    usageFlags = usage;
    m_bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
    m_bufferCreateInfo.usage = usage;
    m_bufferCreateInfo.size = 0;
}
