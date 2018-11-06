//
// Created by Peter Eichinger on 15.10.18.
//

#pragma once
#include <vulkan/vulkan.hpp>
#include "saiga/export.h"
#include "saiga/vulkan/memory/ChunkCreator.h"
#include "saiga/util/imath.h"
#include "saiga/vulkan/memory/MemoryLocation.h"
#include "BaseMemoryAllocator.h"

#include <limits>
#include <saiga/util/easylogging++.h>

using namespace Saiga::Vulkan::Memory;

namespace Saiga{
namespace Vulkan{
namespace Memory{

struct SAIGA_GLOBAL SimpleMemoryAllocator : public BaseMemoryAllocator {
private:
    vk::BufferCreateInfo m_bufferCreateInfo;
    vk::Device m_device;
    vk::PhysicalDevice m_physicalDevice;

    std::vector<MemoryLocation> m_allocations;

    uint32_t findMemoryType(uint32_t typeFilter, const vk::MemoryPropertyFlags &properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = m_physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }


public:
    ~SimpleMemoryAllocator() {
        destroy();
    }
    vk::MemoryPropertyFlags flags;
    vk::BufferUsageFlags  usageFlags;

    SimpleMemoryAllocator() : BaseMemoryAllocator(false) {}

    void init(vk::Device _device, vk::PhysicalDevice _physicalDevice, const vk::MemoryPropertyFlags &_flags,
              const vk::BufferUsageFlags &usage, bool _mapped = false) {
        m_device = _device;
        m_physicalDevice = _physicalDevice;
        flags = _flags;
        mapped = _mapped;
        usageFlags = usage;
        m_bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        m_bufferCreateInfo.usage = usage;
        m_bufferCreateInfo.size = 0;
    }

    MemoryLocation allocate(vk::DeviceSize size) override {
        m_bufferCreateInfo.size = size;
        auto buffer = m_device.createBuffer(m_bufferCreateInfo);

        auto memReqs = m_device.getBufferMemoryRequirements(buffer);

        vk::MemoryAllocateInfo info;
        info.allocationSize = memReqs.size;
        info.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits,flags);
        auto memory = m_device.allocateMemory(info);

        void* mappedPtr = nullptr;
        if (mapped) {
            mappedPtr = m_device.mapMemory(memory,0, memReqs.size);
        }
        m_device.bindBufferMemory(buffer, memory,0);

        m_allocations.emplace_back(buffer, memory, 0,size, mappedPtr);
        return m_allocations.back();
    }

    void destroy() {
        for(auto& location : m_allocations) {
            location.destroy(m_device);
        }
        m_allocations.clear();
    }

    void deallocate(MemoryLocation &location) override {
        LOG(INFO) << "Simple Allocator: Deallocating" << location.memory << std::endl;
        location.destroy(m_device);
        auto newEnd = std::remove(m_allocations.begin(), m_allocations.end(), location);
        m_allocations.erase(newEnd);
    }
};


}
}
}
