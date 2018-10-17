//
// Created by Peter Eichinger on 08.10.18.
//

#pragma once
#include <vulkan/vulkan.hpp>
#include <iostream>

namespace Saiga {
namespace Vulkan {
namespace Memory {

struct Chunk {
    vk::DeviceMemory memory;
    vk::DeviceSize size;
    vk::MemoryPropertyFlags flags;

    Chunk(vk::DeviceMemory mem, vk::DeviceSize memSize, vk::MemoryPropertyFlags memoryFlags) :
        memory(mem), size(memSize), flags(memoryFlags) {
    }
};

}
}
}



