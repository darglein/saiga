/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "Buffer.h"
#include "saiga/vulkan/Base.h"

#include "saiga/vulkan/Vertex.h"
#include "saiga/vulkan/memory/VulkanMemory.h"
namespace Saiga {
namespace Vulkan {


template<typename VertexType>
class SAIGA_TEMPLATE VertexBuffer : public Buffer
{
private:
    MemoryLocation m_memoryLocation;
public:
    int vertexCount;

    void initNew(
            VulkanBase& base,
            int count,
            vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eHostVisible| vk::MemoryPropertyFlagBits::eHostCoherent
            )
    {
        vertexCount = count;
        size_t size = sizeof(VertexType) * vertexCount;
        m_memoryLocation = base.memory.vertexIndexAllocator.allocate(size);
        buffer = m_memoryLocation.buffer;
        DeviceMemory::memory = m_memoryLocation.memory;
    }

    void init(
            VulkanBase& base,
            int count,
            vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eHostVisible| vk::MemoryPropertyFlagBits::eHostCoherent
    )
    {
        vertexCount = count;
        size_t size = sizeof(VertexType) * vertexCount;
//        m_memoryLocation = memory.vertexIndexAllocator.allocate(base,memory.chunkAllocator, size);
//        buffer = m_memoryLocation.buffer;
//        DeviceMemory::memory = m_memoryLocation.memory;
        createBuffer(base,size,vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eTransferDst);
        allocateMemoryBuffer(base,flags);
        m_memoryLocation = {buffer, DeviceMemory::memory, 0};
    }

    void upload(vk::CommandBuffer cmd, const std::vector<VertexType>& vertices)
    {
        vertexCount = vertices.size();
        size_t newSize = sizeof(VertexType) * vertexCount;
        SAIGA_ASSERT(newSize <= size);
        Buffer::upload(cmd,0,newSize,vertices.data());
    }

    void bind(vk::CommandBuffer &cmd)
    {
        cmd.bindVertexBuffers(0, m_memoryLocation.buffer, m_memoryLocation.offset);
    }

    void draw(vk::CommandBuffer &cmd)
    {
        cmd.draw(vertexCount,1,0,0);
    }

    void draw(vk::CommandBuffer &cmd, int count, int first = 0)
    {
        cmd.draw(count,1,first,0);
    }
};

}
}
