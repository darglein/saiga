//
// Created by Peter Eichinger on 2019-02-25.
//

#pragma once
#include "MemoryLocation.h"

namespace Saiga::Vulkan::Memory
{
struct SAIGA_VULKAN_API BufferData
{
    vk::Buffer buffer;

    BufferData(vk::Buffer _buffer) : buffer(_buffer) {}

    BufferData(std::nullptr_t) : buffer(nullptr) {}

    /**
     * Destroys the data that is owned by this object uniquely.
     * @param device device for the data.
     */
    void destroy_owned_data(vk::Device device) {}


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

inline void copy_buffer(vk::CommandBuffer cmd, BufferMemoryLocation* target, BufferMemoryLocation* source)
{
    SafeAccessor safe(*source, *target);
    SAIGA_ASSERT(target->size == source->size, "Different size copies are not supported");
    vk::BufferCopy bc{source->offset, target->offset, target->size};

    cmd.copyBuffer(static_cast<vk::Buffer>(source->data), static_cast<vk::Buffer>(target->data), bc);
}
}  // namespace Saiga::Vulkan::Memory
