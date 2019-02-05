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
namespace Saiga::Vulkan
{
// struct VulkanBase;

namespace Memory
{
struct SAIGA_VULKAN_API MemoryLocation
{
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::DeviceSize offset;
    vk::DeviceSize size;
    void* mappedPointer;

    explicit MemoryLocation(vk::DeviceSize _size)
        : buffer(nullptr), memory(nullptr), offset(0), size(_size), mappedPointer(nullptr)
    {
    }

    explicit MemoryLocation(vk::Buffer _buffer = nullptr, vk::DeviceMemory _memory = nullptr,
                            vk::DeviceSize _offset = 0, vk::DeviceSize _size = 0, void* _basePointer = nullptr)
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

    MemoryLocation& operator=(MemoryLocation&& other) noexcept
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
    inline void make_invalid()
    {
        this->buffer        = nullptr;
        this->memory        = nullptr;
        this->offset        = VK_WHOLE_SIZE;
        this->size          = VK_WHOLE_SIZE;
        this->mappedPointer = nullptr;
    }
    void mappedUpload(vk::Device device, const void* data);


    void mappedDownload(vk::Device device, void* data) const;

   public:
    void copy_to(vk::CommandBuffer cmd, MemoryLocation* target) const;

    void upload(vk::Device device, const void* data);

    void download(vk::Device device, void* data) const;

    void* map(vk::Device device);

    void destroy(const vk::Device& device);

    void* getPointer() const;



    bool operator==(const MemoryLocation& rhs) const
    {
        return std::tie(buffer, memory, offset, size, mappedPointer) ==
               std::tie(rhs.buffer, rhs.memory, rhs.offset, rhs.size, rhs.mappedPointer);
    }

    bool operator!=(const MemoryLocation& rhs) const { return !(rhs == *this); }


    friend std::ostream& operator<<(std::ostream& os, const MemoryLocation& location)
    {
        os << "{" << location.memory << ", " << location.buffer << ", " << location.offset << " " << location.size;

        if (location.mappedPointer)
        {
            os << ", " << location.mappedPointer;
        }

        os << "}";
        return os;
    }
};

}  // namespace Memory
}  // namespace Saiga::Vulkan
