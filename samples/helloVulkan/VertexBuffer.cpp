/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VertexBuffer.h"
#include "vulkanHelper.h"
#include "cube_data.h"

namespace Saiga {
namespace Vulkan {

void VertexBuffer::init(VulkanBase &base)
{
    {
        vk::Result res;

        /*
         * Set up a vertex buffer:
         * - Create a buffer
         * - Map it and write the vertex data into it
         * - Bind it using vkCmdBindVertexBuffers
         * - Later, at pipeline creation,
         * -      fill in vertex input part of the pipeline with relevent data
         */

        vk::BufferCreateInfo buf_info = {};
//        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
//        buf_info.pNext = NULL;
        buf_info.usage = vk::BufferUsageFlagBits::eVertexBuffer;
        buf_info.size = sizeof(g_vb_solid_face_colors_Data);
        buf_info.queueFamilyIndexCount = 0;
        buf_info.pQueueFamilyIndices = NULL;
        buf_info.sharingMode = vk::SharingMode::eExclusive;
//        buf_info.flags = 0;
        res = base.device.createBuffer(&buf_info, NULL, &vertexbuf);

        vk::MemoryRequirements mem_reqs;
        base.device.getBufferMemoryRequirements(vertexbuf, &mem_reqs);

        vk::MemoryAllocateInfo alloc_info = {};
//        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
//        alloc_info.pNext = NULL;
        alloc_info.memoryTypeIndex = 0;

        alloc_info.allocationSize = mem_reqs.size;
        bool pass = Vulkan::memory_type_from_properties(base.memory_properties,mem_reqs.memoryTypeBits,
                                           vk::MemoryPropertyFlagBits::eHostVisible| vk::MemoryPropertyFlagBits::eHostCoherent,
                                           &alloc_info.memoryTypeIndex);

        SAIGA_ASSERT(pass);

        res = base.device.allocateMemory(&alloc_info,nullptr,&vertexmem);

        uint8_t *pData;
        base.device.mapMemory(vertexmem, 0, mem_reqs.size, vk::MemoryMapFlags(), (void **)&pData);
//        assert(res == VK_SUCCESS);
        SAIGA_ASSERT(res == vk::Result::eSuccess);

        memcpy(pData, g_vb_solid_face_colors_Data, sizeof(g_vb_solid_face_colors_Data));

        base.device.unmapMemory(vertexmem);

        base.device.bindBufferMemory(vertexbuf,vertexmem,0);

        /* We won't use these here, but we will need this info when creating the
         * pipeline */
        vi_binding.binding = 0;
        vi_binding.inputRate = vk::VertexInputRate::eVertex;
        vi_binding.stride = sizeof(g_vb_solid_face_colors_Data[0]);

        vi_attribs[0].binding = 0;
        vi_attribs[0].location = 0;
        vi_attribs[0].format = vk::Format::eR32G32B32A32Sfloat;
        vi_attribs[0].offset = 0;
        vi_attribs[1].binding = 0;
        vi_attribs[1].location = 1;
        vi_attribs[1].format = vk::Format::eR32G32B32A32Sfloat;
        vi_attribs[1].offset = 16;
    }
}




}
}
