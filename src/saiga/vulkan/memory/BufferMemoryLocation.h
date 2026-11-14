/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "MemoryLocation.h"

namespace Saiga::Vulkan::Memory
{
struct SAIGA_VULKAN_API BufferData
{
    // Every allocation has its own buffer (bound at offset 0 of the allocation).
    vk::Buffer buffer;
    // The usage the buffer was created with (the defragmentation recreates it with the same usage).
    vk::BufferUsageFlags usage;

    BufferData(vk::Buffer _buffer, vk::BufferUsageFlags _usage = {}) : buffer(_buffer), usage(_usage) {}

    BufferData(std::nullptr_t) : buffer(nullptr) {}

    // Destroys the buffer.
    void destroy_owned_data(vk::Device device)
    {
        if (buffer)
        {
            device.destroy(buffer);
            buffer = nullptr;
        }
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

using BufferMemoryLocation = BaseMemoryLocation<BufferData>;

}  // namespace Saiga::Vulkan::Memory
