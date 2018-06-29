/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VertexBuffer.h"
#include "saiga/vulkan/vulkanHelper.h"

#include "saiga/vulkan/Vertex.h"

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

        static const Saiga::Vulkan::Vertex Triangle[] =
        {
            { {  1.0f,  1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
            { { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
            { {  0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } }
        };


        createBuffer(base,sizeof(Triangle),vk::BufferUsageFlagBits::eVertexBuffer);

        allocateMemory(base);



        upload(base,0,sizeof(Triangle),Triangle);

        base.device.bindBufferMemory(buffer,memory,0);

        VKVertexAttribBinder<Vertex> va;
        va.getVKAttribs(vi_binding,vi_attribs);
    }
}




}
}
