//
// Created by Peter Eichinger on 2019-01-22.
//

#include "MemoryLocation.h"

namespace Saiga::Vulkan::Memory
{
void MemoryLocation::mappedUpload(vk::Device device, const void* data)
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

void MemoryLocation::mappedDownload(vk::Device device, void* data) const
{
    SAIGA_ASSERT(!mappedPointer, "Memory already mapped");
    void* target = device.mapMemory(memory, offset, size);
    std::memcpy(data, target, size);
    device.unmapMemory(memory);
}

void MemoryLocation::upload(vk::Device device, const void* data)
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

void MemoryLocation::download(vk::Device device, void* data) const
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

void* MemoryLocation::map(vk::Device device)
{
    SAIGA_ASSERT(!mappedPointer, "Memory already mapped");
    mappedPointer = device.mapMemory(memory, offset, size);
    return mappedPointer;
}

void MemoryLocation::destroy(const vk::Device& device)
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

void* MemoryLocation::getPointer() const
{
    SAIGA_ASSERT(mappedPointer, "Memory is not mapped");
    return static_cast<char*>(mappedPointer) + offset;
}

void MemoryLocation::copy_to(vk::CommandBuffer cmd, MemoryLocation* target) const
{
    SAIGA_ASSERT(target->size == this->size, "Different size copies are not supported");
    vk::BufferCopy bc{this->offset, target->offset, target->size};

    cmd.copyBuffer(this->buffer, target->buffer, bc);
}

}  // namespace Saiga::Vulkan::Memory