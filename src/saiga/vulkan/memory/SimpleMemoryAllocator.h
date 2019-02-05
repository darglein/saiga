//
// Created by Peter Eichinger on 15.10.18.
//

#pragma once
#include "saiga/export.h"
#include "saiga/core/util/imath.h"
#include "saiga/vulkan/memory/ChunkCreator.h"
#include "saiga/vulkan/memory/MemoryLocation.h"

#include "BaseMemoryAllocator.h"
#include "FindMemoryType.h"
#include "MemoryType.h"
#include "SafeAllocator.h"

#include <limits>
#include <vulkan/vulkan.hpp>

#include <saiga/core/util/easylogging++.h>


namespace Saiga::Vulkan::Memory
{
struct SAIGA_VULKAN_API SimpleMemoryAllocator : public BaseMemoryAllocator
{
   private:
    std::mutex mutex;
    vk::Device m_device;
    vk::PhysicalDevice m_physicalDevice;
    std::vector<std::unique_ptr<MemoryLocation>> m_allocations;
    std::string gui_identifier;

   public:
    BufferType type;

   private:
    vk::BufferCreateInfo m_bufferCreateInfo;

   public:
    SimpleMemoryAllocator(vk::Device _device, vk::PhysicalDevice _physicalDevice, BufferType _type)
        : BaseMemoryAllocator(),
          m_device(_device),
          m_physicalDevice(_physicalDevice),
          type(_type),
          m_bufferCreateInfo(vk::BufferCreateFlags(), 0, _type.usageFlags, vk::SharingMode::eExclusive)
    {
        std::stringstream identifier_stream;
        identifier_stream << "Simple " << type;
        gui_identifier = identifier_stream.str();
    }

    SimpleMemoryAllocator(SimpleMemoryAllocator&& other) noexcept
        : BaseMemoryAllocator(std::move(other)),
          m_device(other.m_device),
          m_physicalDevice(other.m_physicalDevice),
          m_allocations(std::move(other.m_allocations)),
          gui_identifier(std::move(other.gui_identifier)),
          type(other.type),
          m_bufferCreateInfo(std::move(other.m_bufferCreateInfo))
    {
    }

    SimpleMemoryAllocator& operator=(SimpleMemoryAllocator&& other) noexcept
    {
        BaseMemoryAllocator::operator=(std::move(static_cast<BaseMemoryAllocator&&>(other)));
        m_bufferCreateInfo           = std::move(other.m_bufferCreateInfo);
        m_device                     = other.m_device;
        m_physicalDevice             = other.m_physicalDevice;
        m_allocations                = std::move(other.m_allocations);
        gui_identifier               = std::move(other.gui_identifier);
        type                         = other.type;
        return *this;
    }

    ~SimpleMemoryAllocator() override { destroy(); }


    MemoryLocation* allocate(vk::DeviceSize size) override;

    void destroy() override;

    void deallocate(MemoryLocation* location) override;

    void showDetailStats() override;

    MemoryStats collectMemoryStats() override;
};


}  // namespace Saiga::Vulkan::Memory