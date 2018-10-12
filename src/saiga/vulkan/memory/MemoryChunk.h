//
// Created by Peter Eichinger on 08.10.18.
//

#pragma once
#include <vulkan/vulkan.hpp>
#include <iostream>

namespace Saiga {
namespace Vulkan {
namespace Memory {

struct MemoryChunk {
    vk::DeviceMemory memory;
    vk::DeviceSize size;
    vk::MemoryPropertyFlags flags;

    MemoryChunk(vk::DeviceMemory mem, vk::DeviceSize memSize, vk::MemoryPropertyFlags memoryFlags) :
        memory(mem), size(memSize), flags(memoryFlags) {
    }
};

}
}
}



