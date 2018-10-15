//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include <vulkan/vulkan.hpp>
#include "MemoryAllocator.h"
#include "ChunkAllocator.h"
#include "SimpleMemoryAllocator.h"
namespace Saiga {
namespace Vulkan {
namespace Memory {


struct VulkanMemory {
    ChunkAllocator chunkAllocator;
    MemoryAllocator vertexIndexAllocator;
    SimpleMemoryAllocator hostVertexIndexAllocator;
    SimpleMemoryAllocator stagingAllocator;
    SimpleMemoryAllocator uniformAllocator;

    void init(vk::PhysicalDevice _pDevice, vk::Device _device) {
        chunkAllocator.init(_pDevice, _device);
        vertexIndexAllocator.init(_device, &chunkAllocator, vk::MemoryPropertyFlagBits::eDeviceLocal,
                vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst);
        stagingAllocator.init(_device, _pDevice, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                vk::BufferUsageFlagBits::eTransferSrc);
        uniformAllocator.init(_device, _pDevice, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                              vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eUniformBuffer);
        hostVertexIndexAllocator.init(_device,_pDevice,vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                      vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst );
    }


    MemoryAllocatorBase& getAllocator(const vk::BufferUsageFlags &usage, const vk::MemoryPropertyFlags& flags = vk::MemoryPropertyFlagBits::eDeviceLocal) {
        if ((usage & vk::BufferUsageFlagBits::eTransferSrc) == vk::BufferUsageFlagBits::eTransferSrc) {
            return stagingAllocator;
        }

        if ((usage & vk::BufferUsageFlagBits::eUniformBuffer) == vk::BufferUsageFlagBits::eUniformBuffer) {
            return uniformAllocator;
        }
        if ((usage & vk::BufferUsageFlagBits::eVertexBuffer) == vk::BufferUsageFlagBits::eVertexBuffer ||
                (usage & vk::BufferUsageFlagBits::eIndexBuffer) == vk::BufferUsageFlagBits::eIndexBuffer) {
            if ((flags & vk::MemoryPropertyFlagBits::eHostVisible) == vk::MemoryPropertyFlagBits::eHostVisible){
                return hostVertexIndexAllocator;
            }
            return vertexIndexAllocator;
        }
//        if () {
//            return vertexIndexAllocator;
//        }

        throw std::runtime_error("Unknown allocator");
    }


};


}
}
}


