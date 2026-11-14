/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/buffer/Buffer.h"

#include <vector>

namespace Saiga
{
namespace Vulkan
{
/**
 * The draw parameters of indirect draws (vkCmdDrawIndirect / vkCmdDrawIndexedIndirect): the GPU reads the vertex,
 * index and instance ranges from this buffer, so many objects can be drawn with one call and the parameters can be
 * written by a compute shader (the buffer is also a storage buffer).
 *
 * CommandType is vk::DrawIndexedIndirectCommand (with an index buffer bound) or vk::DrawIndirectCommand.
 *
 * The memory is host visible: setCommands() writes directly and must not be called while a frame in flight draws
 * with the buffer.
 *
 * Device features (requested by VulkanRenderer if available): without multiDrawIndirect the commands are drawn one
 * by one, and a firstInstance != 0 needs drawIndirectFirstInstance.
 */
template <typename CommandType = vk::DrawIndexedIndirectCommand>
class SAIGA_TEMPLATE IndirectBuffer : public Buffer
{
    static_assert(std::is_same<CommandType, vk::DrawIndexedIndirectCommand>::value ||
                      std::is_same<CommandType, vk::DrawIndirectCommand>::value,
                  "Only indirect draw commands are allowed");

   public:
    IndirectBuffer()                                     = default;
    IndirectBuffer(IndirectBuffer&& other) noexcept      = default;
    IndirectBuffer& operator=(IndirectBuffer&&) noexcept = default;
    ~IndirectBuffer() override                           = default;

    // The number of commands draw() executes
    uint32_t drawCount = 0;

    void init(VulkanBase& base, uint32_t maxDrawCount)
    {
        capacity  = maxDrawCount;
        drawCount = 0;
        createBuffer(base, sizeof(CommandType) * maxDrawCount,
                     vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }

    void init(VulkanBase& base, const std::vector<CommandType>& commands)
    {
        init(base, (uint32_t)commands.size());
        setCommands(commands);
    }

    void setCommands(const std::vector<CommandType>& commands)
    {
        SAIGA_ASSERT(m_memoryLocation && commands.size() <= capacity, "IndirectBuffer: too many commands");
        drawCount = (uint32_t)commands.size();
        if (drawCount > 0) upload(commands.data(), commands.size() * sizeof(CommandType));
    }

    // Draws the commands [first, first + count), by default all. The pipeline and the vertex (and index) buffers must
    // be bound.
    void draw(vk::CommandBuffer cmd, uint32_t first = 0, uint32_t count = UINT32_MAX)
    {
        SAIGA_ASSERT(m_memoryLocation && first <= drawCount);
        count = std::min(count, drawCount - first);
        if (count == 0) return;

        vk::Buffer buffer;
        {
            Memory::SafeAccessor acc(*m_memoryLocation);
            buffer = m_memoryLocation->data;
        }
        vk::DeviceSize offset = first * sizeof(CommandType);
        if (count == 1 || base->enabledFeatures.multiDrawIndirect)
        {
            drawIndirect(cmd, buffer, offset, count);
        }
        else
        {
            for (uint32_t i = 0; i < count; ++i) drawIndirect(cmd, buffer, offset + i * sizeof(CommandType), 1);
        }
    }

   private:
    uint32_t capacity = 0;

    static void drawIndirect(vk::CommandBuffer cmd, vk::Buffer buffer, vk::DeviceSize offset, uint32_t count)
    {
        if constexpr (std::is_same<CommandType, vk::DrawIndexedIndirectCommand>::value)
            cmd.drawIndexedIndirect(buffer, offset, count, sizeof(CommandType));
        else
            cmd.drawIndirect(buffer, offset, count, sizeof(CommandType));
    }
};

}  // namespace Vulkan
}  // namespace Saiga
