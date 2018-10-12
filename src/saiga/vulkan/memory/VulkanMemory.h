//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include <vulkan/vulkan.hpp>
#include "MemoryAllocator.h"
#include "ChunkAllocator.h"
namespace Saiga {
namespace Vulkan {
namespace Memory {


struct VulkanMemory {
    ChunkAllocator chunkAllocator;
    MemoryAllocator vertexIndexAllocator;

    void init(vk::PhysicalDevice _pDevice, vk::Device _device) {
        chunkAllocator.init(_pDevice, _device);
        vertexIndexAllocator.init(_device, &chunkAllocator, vk::MemoryPropertyFlagBits::eDeviceLocal,
                vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst);
    }
};


}
}
}


