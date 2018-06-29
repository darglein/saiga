/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "Buffer.h"
#include "saiga/vulkan/vulkanBase.h"

#include "saiga/vulkan/Vertex.h"
namespace Saiga {
namespace Vulkan {


template<typename VertexType>
class SAIGA_GLOBAL VertexBuffer : public Buffer
{
public:

    vk::VertexInputBindingDescription vi_binding;
    std::vector<vk::VertexInputAttributeDescription> vi_attribs;

    void init(VulkanBase& base, std::vector<VertexType>& vertices)
    {
        size_t size = sizeof(VertexType) * vertices.size();
        createBuffer(base,size,vk::BufferUsageFlagBits::eVertexBuffer);
        allocateMemory(base);


        DeviceMemory::upload(0,size,vertices.data());
        base.device.bindBufferMemory(buffer,memory,0);
        VKVertexAttribBinder<Vertex> va;
        va.getVKAttribs(vi_binding,vi_attribs);
    }

    void bind(vk::CommandBuffer &cmd, vk::DeviceSize offset = 0)
    {
        cmd.bindVertexBuffers( 0, buffer, offset);
    }
};

}
}
