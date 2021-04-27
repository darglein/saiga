/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "CommandPool.h"

namespace Saiga
{
namespace Vulkan
{
void CommandPool::destroy()
{
    if (device && commandPool) device.destroyCommandPool(commandPool);
}

void CommandPool::create(vk::Device device, std::mutex* _mutex, uint32_t queueFamilyIndex_,
                         vk::CommandPoolCreateFlags flags)
{
    this->device = device;
    this->mutex  = _mutex;
    vk::CommandPoolCreateInfo info(flags, queueFamilyIndex_);

    commandPool = device.createCommandPool(info);
    SAIGA_ASSERT(commandPool);
}

vk::CommandBuffer CommandPool::allocateCommandBuffer(vk::CommandBufferLevel level)
{
    std::scoped_lock lock(*mutex);

    vk::CommandBufferAllocateInfo cmdBufAllocateInfo(commandPool, level, 1);

    vk::CommandBuffer buffer;
    CHECK_VK(device.allocateCommandBuffers(&cmdBufAllocateInfo, &buffer));
    return buffer;
}

std::vector<vk::CommandBuffer> CommandPool::allocateCommandBuffers(uint32_t count, vk::CommandBufferLevel level)
{
    std::scoped_lock lock(*mutex);

    SAIGA_ASSERT(count > 0);
    vk::CommandBufferAllocateInfo cmdBufAllocateInfo(commandPool, level, count);

    std::vector<vk::CommandBuffer> buffers = device.allocateCommandBuffers(cmdBufAllocateInfo);
    SAIGA_ASSERT(buffers.size() == count);
    return buffers;
}

void CommandPool::freeCommandBuffer(vk::CommandBuffer cmd)
{
    std::scoped_lock lock(*mutex);

    device.freeCommandBuffers(commandPool, cmd);
}

void CommandPool::freeCommandBuffers(std::vector<vk::CommandBuffer>& cmds)
{
    std::scoped_lock lock(*mutex);

    if (cmds.empty()) return;
    device.freeCommandBuffers(commandPool, cmds);
}

}  // namespace Vulkan
}  // namespace Saiga
