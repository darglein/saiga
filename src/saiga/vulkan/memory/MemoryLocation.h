//
// Created by Peter Eichinger on 08.10.18.
//

#pragma once
#include "saiga/export.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/vulkan/memory/Chunk.h"

#include <ostream>
#include <saiga/core/util/assert.h>
#include <vulkan/vulkan.hpp>
namespace Saiga
{
namespace Vulkan
{
struct VulkanBase;

namespace Memory
{
struct SAIGA_VULKAN_API MemoryLocation
{
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::DeviceSize offset;
    vk::DeviceSize size;
    void* mappedPointer;

    MemoryLocation() : buffer(nullptr), memory(nullptr), offset(0), size(0), mappedPointer(nullptr) {}

    MemoryLocation(vk::Buffer _buffer, vk::DeviceMemory _memory, vk::DeviceSize _offset, vk::DeviceSize _size = 0,
                   void* _basePointer = nullptr)
        : buffer(_buffer), memory(_memory), offset(_offset), size(_size), mappedPointer(nullptr)
    {
        if (_basePointer)
        {
            mappedPointer = static_cast<char*>(_basePointer) + offset;
        }
    }

    MemoryLocation(const MemoryLocation& other) = default;

    MemoryLocation& operator=(const MemoryLocation& other) = default;

    MemoryLocation(MemoryLocation&& other) noexcept
        : buffer(other.buffer),
          memory(other.memory),
          offset(other.offset),
          size(other.size),
          mappedPointer(other.mappedPointer)
    {
        other.make_invalid();
    }

    MemoryLocation& operator=(MemoryLocation&& other)
    {
        buffer        = other.buffer;
        memory        = other.memory;
        offset        = other.offset;
        size          = other.size;
        mappedPointer = other.mappedPointer;

        other.make_invalid();
        return *this;
    }

    explicit operator bool() { return memory; }

   private:
    void mappedUpload(vk::Device device, const void* data)
    {
        SAIGA_ASSERT(!mappedPointer, "Memory already mapped");
        void* target;
        vk::Result result = device.mapMemory(memory, offset, size, vk::MemoryMapFlags(), &target);
        if (result != vk::Result::eSuccess)
        {
            LOG(FATAL) << "Could not map " << memory << vk::to_string(result);
        }
        //        void *target = result.;
        //        void* target = device.mapMemory(memory, offset,size);
        std::memcpy(target, data, size);
        device.unmapMemory(memory);
    }


    void mappedDownload(vk::Device device, void* data) const
    {
        SAIGA_ASSERT(!mappedPointer, "Memory already mapped");
        void* target = device.mapMemory(memory, offset, size);
        std::memcpy(data, target, size);
        device.unmapMemory(memory);
    }

   public:
    void upload(vk::Device device, const void* data)
    {
        if (mappedPointer)
        {
            std::memcpy(mappedPointer, data, size);
        }
        else
        {
            mappedUpload(device, data);
        }
    }

    void download(vk::Device device, void* data) const
    {
        if (mappedPointer)
        {
            std::memcpy(data, mappedPointer, size);
        }
        else
        {
            mappedDownload(device, data);
        }
    }

    void* map(vk::Device device)
    {
        SAIGA_ASSERT(!mappedPointer, "Memory already mapped");
        mappedPointer = device.mapMemory(memory, offset, size);
        return mappedPointer;
    }

    void unmap(vk::Device device)
    {
        SAIGA_ASSERT(mappedPointer, "Memory not mapped");
        device.unmapMemory(memory);
        mappedPointer = nullptr;
    }

    void destroy(const vk::Device& device)
    {
        SAIGA_ASSERT(memory, "Already destroyed");
        if (buffer)
        {
            device.destroy(buffer);
            buffer = nullptr;
        }
        if (memory)
        {
            device.free(memory);
            memory = nullptr;
        }
        mappedPointer = nullptr;
    }

    void* getPointer() const
    {
        SAIGA_ASSERT(mappedPointer, "Memory is not mapped");
        return static_cast<char*>(mappedPointer) + offset;
    }



    bool operator==(const MemoryLocation& rhs) const
    {
        return std::tie(buffer, memory, offset, size, mappedPointer) ==
               std::tie(rhs.buffer, rhs.memory, rhs.offset, rhs.size, rhs.mappedPointer);
    }

    bool operator!=(const MemoryLocation& rhs) const { return !(rhs == *this); }


    friend std::ostream& operator<<(std::ostream& os, const MemoryLocation& location)
    {
        os << "{" << location.memory << ", " << location.buffer << ", " << location.offset << "-" << location.size;

        if (location.mappedPointer)
        {
            os << ", " << location.mappedPointer;
        }

        os << "}";
        return os;
    }

   public:
    inline void make_invalid()
    {
        this->buffer        = nullptr;
        this->memory        = nullptr;
        this->offset        = VK_WHOLE_SIZE;
        this->size          = VK_WHOLE_SIZE;
        this->mappedPointer = nullptr;
    }
};

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
