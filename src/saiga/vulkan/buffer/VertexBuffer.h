/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/Vertex.h"
#include "saiga/vulkan/memory/VulkanMemory.h"

#include "Buffer.h"
namespace Saiga
{
namespace Vulkan
{
template <typename VertexType>
class SAIGA_TEMPLATE VertexBuffer : public Buffer
{
   public:
    VertexBuffer()                              = default;
    VertexBuffer(const VertexBuffer& other)     = delete;
    VertexBuffer(VertexBuffer&& other) noexcept = default;

    VertexBuffer& operator=(const VertexBuffer& other) = delete;
    VertexBuffer& operator=(VertexBuffer&& other) noexcept = default;

    uint32_t vertexCount;

    void init(VulkanBase& base, uint32_t count, vk::MemoryPropertyFlags flags)
    {
        vertexCount = count;
        size_t size = sizeof(VertexType) * vertexCount;

        createBuffer(base, size, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, flags);
        // m_memoryLocation = base.memory.getAllocator(vk::BufferUsageFlagBits::eVertexBuffer, flags).allocate(size);
        //        buffer = m_memoryLocation.buffer;
        //        DeviceMemory::memory = m_memoryLocation.memory;
    }

    void initDeviceLocal(VulkanBase& base, const std::vector<VertexType>& vertices)
    {
        init(base, vertices.size(), vk::MemoryPropertyFlagBits::eDeviceLocal);
        stagedUpload(base, vertices.size() * sizeof(VertexType), vertices.data());
    }

    void upload(vk::CommandBuffer cmd, const std::vector<VertexType>& vertices)
    {
        vertexCount    = vertices.size();
        size_t newSize = sizeof(VertexType) * vertexCount;
        SAIGA_ASSERT(newSize <= m_memoryLocation->size);
        update(cmd, newSize, vertices.data(), 0);
    }


    void draw(vk::CommandBuffer& cmd) { cmd.draw(vertexCount, 1, 0, 0); }

    void draw(vk::CommandBuffer& cmd, int count, int first = 0) { cmd.draw(count, 1, first, 0); }

    void bind(vk::CommandBuffer& cmd, uint32_t firstBinding = 0)
    {
        SAIGA_ASSERT(m_memoryLocation);
        cmd.bindVertexBuffers(firstBinding, m_memoryLocation->data, m_memoryLocation->offset);
    }
};

}  // namespace Vulkan
}  // namespace Saiga
