//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include <vulkan/vulkan.hpp>
#include "BufferChunkAllocator.h"
#include "ChunkBuilder.h"
#include "SimpleMemoryAllocator.h"
#include "ImageChunkAllocator.h"
#include <sstream>
namespace Saiga {
namespace Vulkan {
namespace Memory {


struct VulkanMemory {

private:
    const vk::ImageUsageFlags default_image = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
    const vk::ImageUsageFlags storage_image = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst;
public:

    ChunkBuilder chunkAllocator;
    BufferChunkAllocator vertexIndexAllocator;
    SimpleMemoryAllocator storageAllocator;
    SimpleMemoryAllocator hostVertexIndexAllocator;
    SimpleMemoryAllocator stagingAllocator;
    BufferChunkAllocator uniformAllocator;
    ImageChunkAllocator imageAllocator;
    FirstFitStrategy strategy;



    void init(vk::PhysicalDevice _pDevice, vk::Device _device) {
        strategy = FirstFitStrategy();
        chunkAllocator.init(_pDevice, _device);
        vertexIndexAllocator = BufferChunkAllocator(_device, &chunkAllocator, vk::MemoryPropertyFlagBits::eDeviceLocal,
                vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                strategy, 64*1024*1024);
        stagingAllocator.init(_device, _pDevice, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                vk::BufferUsageFlagBits::eTransferSrc);
        storageAllocator.init(_device, _pDevice, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,
                vk::BufferUsageFlagBits::eStorageBuffer| vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc);
        uniformAllocator= BufferChunkAllocator(_device, &chunkAllocator, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                              vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eUniformBuffer,strategy,1024*1024,true);
        hostVertexIndexAllocator.init(_device,_pDevice,vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                      vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst );
        imageAllocator = ImageChunkAllocator(_device, &chunkAllocator, vk::MemoryPropertyFlagBits::eDeviceLocal, strategy, 64*1024*1024);
    }


    BaseMemoryAllocator& getAllocator(const vk::BufferUsageFlags &usage, const vk::MemoryPropertyFlags& flags = vk::MemoryPropertyFlagBits::eDeviceLocal) {
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

        if ((usage&vk::BufferUsageFlagBits::eStorageBuffer) == vk::BufferUsageFlagBits::eStorageBuffer) {
            return storageAllocator;
        }
        LOG(ERROR) << "No allocator for " << vk::to_string(usage) << ", " << vk::to_string(flags);
        throw std::runtime_error("No allocator found.");
    }

//
//
    BaseMemoryAllocator& getImageAllocator(const vk::MemoryPropertyFlags& flags = vk::MemoryPropertyFlagBits::eDeviceLocal) {
        if ((flags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal) {
            return imageAllocator;
        }
        LOG(ERROR) << "No allocator for: " << vk::to_string(flags);
        throw std::runtime_error("No allocator found.");
    }


    void destroy() {
        vertexIndexAllocator.destroy();
        chunkAllocator.destroy();
        hostVertexIndexAllocator.destroy();
        stagingAllocator.destroy();
        uniformAllocator.destroy();
    }
};


}
}
}


