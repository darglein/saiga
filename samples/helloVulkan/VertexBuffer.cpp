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
        /*
         * Set up a vertex buffer:
         * - Create a buffer
         * - Map it and write the vertex data into it
         * - Bind it using vkCmdBindVertexBuffers
         * - Later, at pipeline creation,
         * -      fill in vertex input part of the pipeline with relevent data
         */


        createBuffer(base,sizeof(g_vb_solid_face_colors_Data),vk::BufferUsageFlagBits::eVertexBuffer);

        allocateMemory(base);



        upload(base,0,sizeof(g_vb_solid_face_colors_Data),g_vb_solid_face_colors_Data);

        base.device.bindBufferMemory(buffer,memory,0);

        VKVertexAttribBinder<Vertex> va;
        va.getVKAttribs(vi_binding,vi_attribs);
    }
}




}
}
