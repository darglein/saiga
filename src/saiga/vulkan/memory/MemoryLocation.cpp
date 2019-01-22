//
// Created by Peter Eichinger on 2019-01-22.
//

#include "MemoryLocation.h"

void Saiga::Vulkan::Memory::MemoryLocation::mappedUpload(vk::Device device, const void *data) {
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

void Saiga::Vulkan::Memory::MemoryLocation::mappedDownload(vk::Device device, void *data) const {
    SAIGA_ASSERT(!mappedPointer, "Memory already mapped");
    void* target = device.mapMemory(memory, offset, size);
    std::memcpy(data, target, size);
    device.unmapMemory(memory);
}

void Saiga::Vulkan::Memory::MemoryLocation::upload(vk::Device device, const void *data) {
    if (mappedPointer)
    {
        std::memcpy(mappedPointer, data, size);
    }
    else
    {
        mappedUpload(device, data);
    }
}

void Saiga::Vulkan::Memory::MemoryLocation::download(vk::Device device, void *data) const {
    if (mappedPointer)
    {
        std::memcpy(data, mappedPointer, size);
    }
    else
    {
        mappedDownload(device, data);
    }
}

void *Saiga::Vulkan::Memory::MemoryLocation::map(vk::Device device) {
    SAIGA_ASSERT(!mappedPointer, "Memory already mapped");
    mappedPointer = device.mapMemory(memory, offset, size);
    return mappedPointer;
}

void Saiga::Vulkan::Memory::MemoryLocation::destroy(const vk::Device &device) {
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

void *Saiga::Vulkan::Memory::MemoryLocation::getPointer() const {
    SAIGA_ASSERT(mappedPointer, "Memory is not mapped");
    return static_cast<char*>(mappedPointer) + offset;
}

void Saiga::Vulkan::Memory::MemoryLocation::make_invalid() {
    this->buffer        = nullptr;
    this->memory        = nullptr;
    this->offset        = VK_WHOLE_SIZE;
    this->size          = VK_WHOLE_SIZE;
    this->mappedPointer = nullptr;
}
