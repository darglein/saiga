/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "UniformBuffer.h"
#include "vulkanHelper.h"

namespace Saiga {
namespace Vulkan {

void UniformBuffer::init(VulkanBase &base)
{
    vk::Result res;
    Projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    View = glm::lookAt(glm::vec3(-5, 3, -10),  // Camera is at (-5,3,-10), in World Space
                            glm::vec3(0, 0, 0),     // and looks at the origin
                            glm::vec3(0, -1, 0)     // Head is up (set to 0,-1,0 to look upside-down)
                            );
    Model = glm::mat4(1.0f);

    // Vulkan clip space has inverted Y and half Z.
    // clang-format off
    Clip = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f,-1.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.5f, 0.0f,
                          0.0f, 0.0f, 0.5f, 1.0f);
    // clang-format on
    MVP = Clip * Projection * View * Model;

    vk::BufferCreateInfo buf_info = {};

    buf_info.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    buf_info.size = sizeof(MVP);
    buf_info.queueFamilyIndexCount = 0;
    buf_info.pQueueFamilyIndices = NULL;
    buf_info.sharingMode = vk::SharingMode::eExclusive;
//    buf_info.flags = 0;
//    res = vkCreateBuffer(info.device, &buf_info, NULL, &info.uniform_data.buf);
    res = base.device.createBuffer(&buf_info,nullptr,&uniformbuf);
    SAIGA_ASSERT(res == vk::Result::eSuccess);

    vk::MemoryRequirements mem_reqs;
//    vkGetBufferMemoryRequirements(info.device, info.uniform_data.buf, &mem_reqs);
    base.device.getBufferMemoryRequirements(uniformbuf,&mem_reqs);

    vk::MemoryAllocateInfo alloc_info = {};
    alloc_info.memoryTypeIndex = 0;
    alloc_info.allocationSize = mem_reqs.size;

    auto pass = memory_type_from_properties(base.memory_properties, mem_reqs.memoryTypeBits,
                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                       &alloc_info.memoryTypeIndex);

    SAIGA_ASSERT(pass);

//    res = vkAllocateMemory(info.device, &alloc_info, NULL, &(info.uniform_data.mem));
    res  = base.device.allocateMemory(&alloc_info,nullptr,&uniformmem);
    SAIGA_ASSERT(res == vk::Result::eSuccess);
//    assert(res == VK_SUCCESS);

    uint8_t *pData;
//    res = vkMapMemory(info.device, info.uniform_data.mem, 0, mem_reqs.size, 0, (void **)&pData);
    res = base.device.mapMemory(uniformmem,0,mem_reqs.size, vk::MemoryMapFlags(), (void **)&pData);
//    assert(res == VK_SUCCESS);
    SAIGA_ASSERT(res == vk::Result::eSuccess);


    memcpy(pData, &MVP, sizeof(MVP));

    base.device.unmapMemory(uniformmem);


//    res = vkBindBufferMemory(info.device, info.uniform_data.buf, info.uniform_data.mem, 0);
    base.device.bindBufferMemory(uniformbuf,uniformmem,0);


    uniformbuffer_info.buffer = uniformbuf;
    uniformbuffer_info.offset = 0;
    uniformbuffer_info.range = sizeof(MVP);

}




}
}
