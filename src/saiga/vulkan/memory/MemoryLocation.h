//
// Created by Peter Eichinger on 08.10.18.
//

#pragma once
#include "saiga/core/util/easylogging++.h"
#include "saiga/export.h"
#include "saiga/vulkan/memory/Chunk.h"

#include <ostream>
#include <saiga/core/util/assert.h>
#include <vulkan/vulkan.hpp>
namespace Saiga::Vulkan::Memory
{
template <typename Data>
struct SAIGA_VULKAN_API BaseMemoryLocation
{
   public:
    Data data;
    vk::DeviceMemory memory;
    vk::DeviceSize offset;
    vk::DeviceSize size;
    void* mappedPointer;

   private:
    bool static_mem;

   public:
    explicit BaseMemoryLocation(vk::DeviceSize _size)
        : data(nullptr), memory(nullptr), offset(0), size(_size), mappedPointer(nullptr), static_mem(true)
    {
    }

    explicit BaseMemoryLocation(Data _data = nullptr, vk::DeviceMemory _memory = nullptr, vk::DeviceSize _offset = 0,
                                vk::DeviceSize _size = 0, void* _basePointer = nullptr)
        : data(_data), memory(_memory), offset(_offset), size(_size), mappedPointer(nullptr), static_mem(true)
    {
        if (_basePointer)
        {
            mappedPointer = static_cast<char*>(_basePointer) + offset;
        }
    }

    BaseMemoryLocation(const BaseMemoryLocation& other) = default;

    BaseMemoryLocation& operator=(const BaseMemoryLocation& other) = default;

    BaseMemoryLocation(BaseMemoryLocation&& other) noexcept
        : data(other.data),
          memory(other.memory),
          offset(other.offset),
          size(other.size),
          mappedPointer(other.mappedPointer),
          static_mem(other.static_mem)
    {
        other.make_invalid();
    }

    BaseMemoryLocation& operator=(BaseMemoryLocation&& other) noexcept
    {
        if (this != &other)
        {
            data          = other.data;
            memory        = other.memory;
            offset        = other.offset;
            size          = other.size;
            mappedPointer = other.mappedPointer;
            static_mem    = other.static_mem;

            other.make_invalid();
        }
        return *this;
    }

    explicit operator bool() { return memory; }

   private:
    inline void make_invalid()
    {
        this->data          = nullptr;
        this->memory        = nullptr;
        this->offset        = VK_WHOLE_SIZE;
        this->size          = VK_WHOLE_SIZE;
        this->mappedPointer = nullptr;
    }
    void mappedUpload(vk::Device device, const void* data)
    {
        SAIGA_ASSERT(!mappedPointer, "Memory already mapped");
        void* target;
        vk::Result result = device.mapMemory(memory, offset, size, vk::MemoryMapFlags(), &target);
        if (result != vk::Result::eSuccess)
        {
            LOG(FATAL) << "Could not map " << memory << vk::to_string(result);
        }
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
    inline bool is_dynamic() const { return !is_static(); }
    inline bool is_static() const { return static_mem; }

    inline void mark_dynamic() { static_mem = false; }

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

    void destroy(const vk::Device& device)
    {
        SAIGA_ASSERT(memory, "Already destroyed");
        if (data)
        {
            data.destroy(device);
            data = nullptr;
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



    bool operator==(const BaseMemoryLocation& rhs) const
    {
        return std::tie(data, memory, offset, size, mappedPointer) ==
               std::tie(rhs.data, rhs.memory, rhs.offset, rhs.size, rhs.mappedPointer);
    }

    bool operator!=(const BaseMemoryLocation& rhs) const { return !(rhs == *this); }


    friend std::ostream& operator<<(std::ostream& os, const BaseMemoryLocation& location)
    {
        os << "{" << location.memory << ", " << location.data << ", " << location.offset << " " << location.size;

        if (location.mappedPointer)
        {
            os << ", " << location.mappedPointer;
        }

        os << "}";
        return os;
    }
};

struct SAIGA_VULKAN_API BufferData
{
    vk::Buffer buffer;

    BufferData(vk::Buffer _buffer) : buffer(_buffer) {}

    BufferData(nullptr_t) : buffer(nullptr) {}

    void destroy(vk::Device device)
    {
        device.destroy(buffer);
        buffer = nullptr;
    }


    operator bool() { return buffer; }

    operator vk::Buffer() { return buffer; }

    operator vk::ArrayProxy<const vk::Buffer>() { return vk::ArrayProxy<const vk::Buffer>(buffer); }

    friend std::ostream& operator<<(std::ostream& os, const BufferData& bufferData)
    {
        os << bufferData.buffer;
        return os;
    }
};



using MemoryLocation = BaseMemoryLocation<BufferData>;


inline void copy_buffer(vk::CommandBuffer cmd, MemoryLocation* target, MemoryLocation* source)
{
    SAIGA_ASSERT(target->size == source->size, "Different size copies are not supported");
    vk::BufferCopy bc{source->offset, target->offset, target->size};

    cmd.copyBuffer(static_cast<vk::Buffer>(source->data), static_cast<vk::Buffer>(target->data), bc);
}
// class MemoryLocation : public BaseMemoryLocation<vk::Buffer>
//{
//};

}  // namespace Saiga::Vulkan::Memory