/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "CommandPool.h"
#include "saiga/vulkan/vulkanHelper.h"

namespace Saiga {
namespace Vulkan {

CommandPool::CommandPool()
{

}

CommandPool::~CommandPool()
{

}

void CommandPool::create(VulkanBase &base, uint32_t queueFamilyIndex, vk::CommandPoolCreateFlags flags)
{
    vk::CommandPoolCreateInfo cmd_pool_info = {};
    cmd_pool_info.queueFamilyIndex = queueFamilyIndex;
    cmd_pool_info.flags = flags;
    CHECK_VK(base.device.createCommandPool(&cmd_pool_info, nullptr, &cmd_pool));
}

vk::CommandBuffer CommandPool::createCommandBuffer(VulkanBase &base, vk::CommandBufferLevel level)
{
    vk::CommandBufferAllocateInfo cmd_info = {};
    cmd_info.commandPool = cmd_pool;
    cmd_info.level = level;
    cmd_info.commandBufferCount = 1;

    vk::CommandBuffer cmd;
    CHECK_VK(base.device.allocateCommandBuffers(&cmd_info,&cmd));
    return cmd;
}

}
}
