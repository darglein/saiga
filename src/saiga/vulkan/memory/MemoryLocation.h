#include <utility>

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

    inline void destroy_data(const vk::Device& device)
    {
        if (data)
        {
            data.destroy(device);
            data = nullptr;
        }
    }

    void destroy(const vk::Device& device)
    {
        SAIGA_ASSERT(memory, "Already destroyed");
        destroy_data(device);
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


    operator bool() const { return buffer; }

    operator vk::Buffer() const { return buffer; }

    operator vk::ArrayProxy<const vk::Buffer>() const { return vk::ArrayProxy<const vk::Buffer>(buffer); }

    bool operator==(const BufferData& other) const { return buffer == other.buffer; }

    friend std::ostream& operator<<(std::ostream& os, const BufferData& bufferData)
    {
        std::stringstream ss;
        ss << std::hex << bufferData.buffer;
        os << ss.str();
        return os;
    }
};

struct SAIGA_VULKAN_API ImageData
{
    vk::ImageLayout layout;
    vk::Image image;
    vk::ImageCreateInfo image_create_info;
    vk::ImageView view;
    vk::ImageViewCreateInfo view_create_info;
    vk::MemoryRequirements image_requirements;

    ImageData(vk::ImageCreateInfo _image_create_info, vk::ImageViewCreateInfo _view_create_info,
              vk::ImageLayout _layout)
        : layout(_layout),
          image(nullptr),
          image_create_info(std::move(_image_create_info)),
          view(nullptr),
          view_create_info(std::move(_view_create_info)),
          image_requirements()
    {
    }

    ImageData(nullptr_t)
        : layout(vk::ImageLayout::eUndefined),
          image(nullptr),
          image_create_info(),
          view(nullptr),
          view_create_info(),
          image_requirements()
    {
    }

    ImageData(const ImageData& other) = default;
    ImageData(ImageData&& other)      = default;

    ImageData& operator=(const ImageData& other) = default;
    ImageData& operator=(ImageData&& other) = default;

    explicit operator bool() const { return image; }

    void copy_create_info_from(ImageData const& other) { set_info(other.image_create_info, other.view_create_info); }

    void set_info(vk::ImageCreateInfo const& _image_create_info, vk::ImageViewCreateInfo const& _view_create_info)
    {
        image_create_info = _image_create_info;
        view_create_info  = _view_create_info;
    }


    void create_image(vk::Device device)
    {
        image                  = device.createImage(image_create_info);
        image_requirements     = device.getImageMemoryRequirements(image);
        view_create_info.image = image;
    }

    void create_view(vk::Device device) { view = device.createImageView(view_create_info); }

    void destroy(vk::Device device)
    {
        if (view)
        {
            device.destroy(view);
        }

        if (image)
        {
            device.destroy(image);
        }

        view   = nullptr;
        image  = nullptr;
        layout = vk::ImageLayout::eUndefined;
    }

    friend std::ostream& operator<<(std::ostream& os, const ImageData& data)
    {
        std::stringstream ss;
        ss << std::hex << vk::to_string(data.layout) << " " << data.image << ", " << data.view;
        os << ss.str();
        return os;
    }
};



using BufferMemoryLocation = BaseMemoryLocation<BufferData>;
using ImageMemoryLocation  = BaseMemoryLocation<ImageData>;

inline void copy_buffer(vk::CommandBuffer cmd, BufferMemoryLocation* target, BufferMemoryLocation* source)
{
    SAIGA_ASSERT(target->size == source->size, "Different size copies are not supported");
    vk::BufferCopy bc{source->offset, target->offset, target->size};

    cmd.copyBuffer(static_cast<vk::Buffer>(source->data), static_cast<vk::Buffer>(target->data), bc);
}

inline void bind_image_data(vk::Device device, ImageMemoryLocation* location, ImageData& data)
{
    device.bindImageMemory(data.image, location->memory, location->offset);
    location->data = data;
}

inline void copy_image(vk::CommandBuffer cmd, ImageMemoryLocation* target, ImageMemoryLocation* source)
{
    SAIGA_ASSERT(target->size == source->size, "Different size copies are not supported");
    const auto& src_data = source->data;
    const auto& dst_data = target->data;
    SAIGA_ASSERT(src_data.image_create_info.mipLevels == dst_data.image_create_info.mipLevels,
                 "Source and Target must have same number of mip levels.");
    SAIGA_ASSERT(src_data.image_create_info.extent == dst_data.image_create_info.extent,
                 "Images must have the same extent");
    SAIGA_ASSERT(src_data.layout == dst_data.layout, "Layouts must be the same");

    static const vk::ImageAspectFlags copy_aspects =
        vk::ImageAspectFlagBits::eColor | vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;

    for (uint32_t mip = 0; mip < src_data.image_create_info.mipLevels; ++mip)
    {
        vk::ImageSubresourceLayers layers{copy_aspects, mip, 0, src_data.image_create_info.arrayLayers};
        vk::ImageCopy ic{layers, vk::Offset3D{0, 0, 0}, layers, vk::Offset3D{0, 0, 0},
                         src_data.image_create_info.extent};

        cmd.copyImage(src_data.image, src_data.layout, dst_data.image, dst_data.layout, ic);
    }
}
}  // namespace Saiga::Vulkan::Memory