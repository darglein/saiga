/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/svulkan.h"

#include <mutex>
namespace Saiga
{
namespace Vulkan
{
/**
 * Wrapper class for vk::CommandPool.
 */
class SAIGA_VULKAN_API CommandPool
{
   public:
    /*
    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
        specifies that command buffers allocated from the pool will be short-lived,
        meaning that they will be reset or freed in a relatively short timeframe.
        This flag may be used by the implementation to control memory allocation behavior within the pool.

    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        allows any command buffer allocated from a pool to be individually reset to the initial state;
        either by calling vkResetCommandBuffer, or via the implicit reset when calling vkBeginCommandBuffer.
        If this flag is not set on a pool, then vkResetCommandBuffer must not be called for
        any command buffer allocated from that pool.

    VK_COMMAND_POOL_CREATE_PROTECTED_BIT
        specifies that command buffers allocated from the pool are protected command buffers.
        If the protected memory feature is not enabled, the VK_COMMAND_POOL_CREATE_PROTECTED_BIT
        bit of flags must not be set.
    */
    void create(vk::Device device, std::mutex* _mutex, uint32_t queueFamilyIndex_ = 0,
                vk::CommandPoolCreateFlags flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    /**
     * Allocate a single command buffer from this pool.
     * If you want to allocate multiple command buffers at once use the function below.
     */
    vk::CommandBuffer allocateCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);

    vk::CommandBuffer createAndBeginOneTimeBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary)
    {
        auto cmd = allocateCommandBuffer(level);
        cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        return cmd;
    }

    /**
     * Allocate an array of command buffers.
     */
    std::vector<vk::CommandBuffer> allocateCommandBuffers(
        uint32_t count, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);


    void freeCommandBuffer(vk::CommandBuffer cmd);
    void freeCommandBuffers(std::vector<vk::CommandBuffer>& cmds);


    explicit operator vk::CommandPool() { return commandPool; }

    explicit operator VkCommandPool() { return commandPool; }

    /**
     * Free the underlying Vulkan objects.
     */
    void destroy();

   private:
    std::mutex* mutex;
    vk::Device device;
    vk::CommandPool commandPool;
};


}  // namespace Vulkan
}  // namespace Saiga
