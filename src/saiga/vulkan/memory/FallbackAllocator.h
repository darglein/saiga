//
// Created by Peter Eichinger on 2018-12-18.
//

#pragma once

#include "saiga/export.h"
#include "saiga/util/imath.h"
#include "saiga/vulkan/memory/ChunkCreator.h"
#include "saiga/vulkan/memory/MemoryLocation.h"

#include "BaseMemoryAllocator.h"
#include "FindMemoryType.h"
#include "MemoryType.h"
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
class FallbackAllocator : public BaseMemoryAllocator
{
   private:
    std::mutex mutex;
    vk::Device m_device;
    vk::PhysicalDevice m_physicalDevice;
    std::vector<std::unique_ptr<MemoryLocation>> m_allocations;
    std::string gui_identifier;

   public:
    FallbackAllocator(vk::Device _device, vk::PhysicalDevice _physicalDevice)
        : BaseMemoryAllocator(), m_device(_device), m_physicalDevice(_physicalDevice)
    {
        std::stringstream identifier_stream;
        identifier_stream << "Fallback allocator";
        gui_identifier = identifier_stream.str();
    }

    FallbackAllocator(FallbackAllocator&& other) noexcept
        : BaseMemoryAllocator(std::move(other)),
          m_device(other.m_device),
          m_physicalDevice(other.m_physicalDevice),
          m_allocations(std::move(other.m_allocations)),
          gui_identifier(std::move(other.gui_identifier))
    {
    }

    FallbackAllocator& operator=(FallbackAllocator&& other) noexcept
    {
        BaseMemoryAllocator::operator=(std::move(static_cast<BaseMemoryAllocator&&>(other)));
        m_device                     = other.m_device;
        m_physicalDevice             = other.m_physicalDevice;
        m_allocations                = std::move(other.m_allocations);
        gui_identifier               = std::move(other.gui_identifier);
        return *this;
    }

    ~FallbackAllocator() override { destroy(); }

    MemoryLocation* allocate(vk::DeviceSize size) override;

    MemoryLocation* allocate(const BufferType& type, vk::DeviceSize size);
    MemoryLocation* allocate(const ImageType& type, const vk::Image& image);

    void destroy() override;

    void deallocate(MemoryLocation* location) override;

    void showDetailStats() override;

    MemoryStats collectMemoryStats() override;
};

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga