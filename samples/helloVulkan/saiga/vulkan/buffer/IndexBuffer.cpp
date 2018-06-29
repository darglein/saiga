/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "IndexBuffer.h"
#include "saiga/vulkan/vulkanHelper.h"


namespace Saiga {
namespace Vulkan {

void IndexBuffer::init(VulkanBase &base)
{

    // Setup indices data
        std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
//        indices.count = static_cast<uint32_t>(indexBuffer.size());
    uint32_t indexBufferSize = indexBuffer.size() * sizeof(uint32_t);



    createBuffer(base,indexBufferSize,vk::BufferUsageFlagBits::eIndexBuffer);

    allocateMemory(base);



    upload(base,0,indexBufferSize,indexBuffer.data());

    base.device.bindBufferMemory(buffer,memory,0);

 }




}
}
