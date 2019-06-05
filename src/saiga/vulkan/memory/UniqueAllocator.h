//
// Created by Peter Eichinger on 2018-12-18.
//

#pragma once

#include "saiga/core/math/imath.h"
#include "saiga/export.h"

#include "BufferMemoryLocation.h"
#include "FindMemoryType.h"
#include "ImageMemoryLocation.h"
#include "MemoryStats.h"
#include "MemoryType.h"
#include "SafeAllocator.h"

#include <limits>
#include <vulkan/vulkan.hpp>

#include <saiga/core/util/easylogging++.h>

namespace Saiga::Vulkan::Memory
{
class SAIGA_VULKAN_API UniqueAllocator
{
   private:
    std::mutex mutex;
    vk::Device m_device;
    vk::PhysicalDevice m_physicalDevice;
    std::vector<std::unique_ptr<BufferMemoryLocation>> m_allocations;
    std::vector<std::unique_ptr<ImageMemoryLocation>> m_image_allocations;
    std::string gui_identifier;

    void destroy(const vk::Device& device, BufferMemoryLocation* memory_location);

    void destroy(const vk::Device& device, ImageMemoryLocation* memory_location);

   public:
    UniqueAllocator(vk::Device _device, vk::PhysicalDevice _physicalDevice)
        : m_device(_device), m_physicalDevice(_physicalDevice)
    {
        std::stringstream identifier_stream;
        identifier_stream << "Fallback allocator";
        gui_identifier = identifier_stream.str();
    }

    UniqueAllocator(UniqueAllocator&& other) noexcept
        : m_device(other.m_device),
          m_physicalDevice(other.m_physicalDevice),
          m_allocations(std::move(other.m_allocations)),
          m_image_allocations(std::move(other.m_image_allocations)),
          gui_identifier(std::move(other.gui_identifier))
    {
    }

    UniqueAllocator& operator=(UniqueAllocator&& other) noexcept
    {
        m_device            = other.m_device;
        m_physicalDevice    = other.m_physicalDevice;
        m_allocations       = std::move(other.m_allocations);
        m_image_allocations = std::move(other.m_image_allocations);
        gui_identifier      = std::move(other.gui_identifier);
        return *this;
    }

    ~UniqueAllocator() { destroy(); }

    UniqueAllocator(const UniqueAllocator&) = delete;
    UniqueAllocator& operator=(const UniqueAllocator&) = delete;

    BufferMemoryLocation* allocate(const BufferType& type, vk::DeviceSize size);
    ImageMemoryLocation* allocate(const ImageType& type, ImageData& image_data);


    void deallocate(BufferMemoryLocation* location);
    void deallocate(ImageMemoryLocation* location);

    void destroy();

    void showDetailStats();

    MemoryStats collectMemoryStats();
};

}  // namespace Saiga::Vulkan::Memory
