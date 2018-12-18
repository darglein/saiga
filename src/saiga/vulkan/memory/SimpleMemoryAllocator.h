//
// Created by Peter Eichinger on 15.10.18.
//

#pragma once
#include "saiga/export.h"
#include "saiga/util/imath.h"
#include "saiga/vulkan/memory/ChunkCreator.h"
#include "saiga/vulkan/memory/MemoryLocation.h"

#include "BaseMemoryAllocator.h"
#include "FindMemoryType.h"
#include "SafeAllocator.h"

#include <limits>
#include <vulkan/vulkan.hpp>

#include <saiga/util/easylogging++.h>


namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
struct SAIGA_GLOBAL SimpleMemoryAllocator : public BaseMemoryAllocator
{
   private:
    std::mutex mutex;
    vk::Device m_device;
    vk::PhysicalDevice m_physicalDevice;
    std::vector<MemoryLocation> m_allocations;
    std::string gui_identifier;

   public:
    vk::MemoryPropertyFlags flags;
    vk::BufferUsageFlags usageFlags;

   private:
    vk::BufferCreateInfo m_bufferCreateInfo;

   public:
    SimpleMemoryAllocator(vk::Device _device, vk::PhysicalDevice _physicalDevice, const vk::MemoryPropertyFlags& _flags,
                          const vk::BufferUsageFlags& usage, bool _mapped = false)
        : BaseMemoryAllocator(_mapped),
          m_device(_device),
          m_physicalDevice(_physicalDevice),
          flags(_flags),
          usageFlags(usage),
          m_bufferCreateInfo(vk::BufferCreateFlags(), 0, usage, vk::SharingMode::eExclusive)
    {
        std::stringstream identifier_stream;
        identifier_stream << "Simple " << vk::to_string(usageFlags) << " " << vk::to_string(flags);
        gui_identifier = identifier_stream.str();
    }

    SimpleMemoryAllocator(SimpleMemoryAllocator&& other) noexcept
        : BaseMemoryAllocator(std::move(other)),
          m_device(other.m_device),
          m_physicalDevice(other.m_physicalDevice),
          m_allocations(std::move(other.m_allocations)),
          gui_identifier(std::move(other.gui_identifier)),
          flags(other.flags),
          usageFlags(other.usageFlags),
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
        flags                        = other.flags;
        usageFlags                   = other.usageFlags;
        return *this;
    }

    ~SimpleMemoryAllocator() override { destroy(); }


    MemoryLocation allocate(vk::DeviceSize size) override;

    void destroy() override;

    void deallocate(MemoryLocation& location) override;

    void showDetailStats() override;

    MemoryStats collectMemoryStats() override;
};


}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
