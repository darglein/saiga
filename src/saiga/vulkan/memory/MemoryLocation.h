//
// Created by Peter Eichinger on 08.10.18.
//

#pragma once
#include <vulkan/vulkan.hpp>
#include <saiga/export.h>
namespace Saiga{
namespace Vulkan{
namespace Memory{

struct SAIGA_GLOBAL MemoryLocation {
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::DeviceSize offset;

    MemoryLocation() : buffer(nullptr), memory(nullptr), offset(0){}
    MemoryLocation(vk::Buffer _buffer, vk::DeviceMemory _memory, vk::DeviceSize _offset) :
        buffer(_buffer), memory(_memory), offset(_offset) { }
};

}
}
}