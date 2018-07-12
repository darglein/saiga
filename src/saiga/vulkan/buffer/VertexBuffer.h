/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "Buffer.h"
#include "saiga/vulkan/Base.h"

#include "saiga/vulkan/Vertex.h"
namespace Saiga {
namespace Vulkan {


template<typename VertexType>
class SAIGA_GLOBAL VertexBuffer : public Buffer
{
public:
    int vertexCount;

    void init(
            VulkanBase& base,
            int count,
            vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eHostVisible| vk::MemoryPropertyFlagBits::eHostCoherent
            )
    {
        vertexCount = count;
        size_t size = sizeof(VertexType) * vertexCount;
        createBuffer(base,size,vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eTransferDst);
        allocateMemoryBuffer(base,flags);
    }

    void init(VulkanBase& base, const std::vector<VertexType>& vertices)
    {
        vertexCount = vertices.size();
        size_t size = sizeof(VertexType) * vertexCount;
        createBuffer(base,size,vk::BufferUsageFlagBits::eVertexBuffer);
        allocateMemoryBuffer(base);
        DeviceMemory::mappedUpload(0,size,vertices.data());
    }

    void bind(vk::CommandBuffer &cmd, vk::DeviceSize offset = 0)
    {
        cmd.bindVertexBuffers( 0, buffer, offset);
    }

    void draw(vk::CommandBuffer &cmd)
    {
        cmd.draw(vertexCount,1,0,0);
    }
};

}
}
