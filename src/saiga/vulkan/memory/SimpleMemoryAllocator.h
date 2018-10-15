//
// Created by Peter Eichinger on 15.10.18.
//

#pragma once
#include <vulkan/vulkan.hpp>
#include "saiga/export.h"
#include "saiga/vulkan/memory/ChunkAllocator.h"
#include "saiga/util/imath.h"
#include "saiga/vulkan/memory/MemoryLocation.h"
#include "MemoryAllocatorBase.h"

#include <limits>
using namespace Saiga::Vulkan::Memory;

namespace Saiga{
namespace Vulkan{
namespace Memory{

struct SAIGA_GLOBAL SimpleMemoryAllocator : public MemoryAllocatorBase {
private:
    vk::BufferCreateInfo m_bufferCreateInfo;
    vk::Device m_device;
    vk::PhysicalDevice m_physicalDevice;

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
    vk::MemoryPropertyFlags flags;
    vk::BufferUsageFlags  usageFlags;
    void init(vk::Device _device, vk::PhysicalDevice _physicalDevice, const vk::MemoryPropertyFlags &_flags,
              const vk::BufferUsageFlags &usage) {
        m_device = _device;
        m_physicalDevice = _physicalDevice;
        flags = _flags;
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

        m_device.bindBufferMemory(buffer, memory,0);
        return {buffer, memory, 0,size};
    }

};


}
}
}
