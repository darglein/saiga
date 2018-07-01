/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "vulkanBase.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL DeviceMemory
{
public:
    vk::Device device;
    size_t size;
    vk::DeviceMemory memory;

    DeviceMemory(){}
    ~DeviceMemory();

    void allocateMemory(VulkanBase &base, const vk::MemoryRequirements& mem_reqs, vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eHostVisible| vk::MemoryPropertyFlagBits::eHostCoherent);

    uint8_t* map(size_t offset, size_t size);
    void unmap();

    void upload(size_t offset, size_t size, const void* data);

//    void upload(VulkanBase &base, size_t offset, size_t size, const void* data);


};

}
}
