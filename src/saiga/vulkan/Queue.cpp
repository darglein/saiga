/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Queue.h"

#include "saiga/core/util/easylogging++.h"

namespace Saiga
{
namespace Vulkan
{
void Queue::create(vk::Device _device, uint32_t _queueFamilyIndex, uint32_t _queueIndex,
                   vk::CommandPoolCreateFlags commandPoolCreateFlags)
{
    device           = _device;
    queueFamilyIndex = _queueFamilyIndex;
    queueIndex       = _queueIndex;
    queue            = _device.getQueue(_queueFamilyIndex, _queueIndex);
    SAIGA_ASSERT(queue);
    commandPool.create(device, &commandPoolCreationMutex, _queueFamilyIndex, commandPoolCreateFlags);
}

void Queue::waitIdle()
{
    VLOG(3) << "Queue: wait idle...";
    submitMutex.lock();
    queue.waitIdle();
    submitMutex.unlock();
    VLOG(3) << "Queue: wait idle done.";
}

void Queue::destroy()
{
    waitIdle();

    VLOG(3) << "Destroying queue with " << commandPools.size() << " command pools";
    for (auto& pool : commandPools)
    {
        VLOG(3) << "Destroying command pool: " << static_cast<vk::CommandPool>(pool);
        pool.destroy();
    }

    commandPool.destroy();

    //    device.destroy(*this);
}

void Queue::submitAndWait(vk::CommandBuffer cmd)
{
    SAIGA_ASSERT(cmd, "invalid command buffer provided");
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    vk::FenceCreateInfo fci{};
    vk::Fence fence = device.createFence(fci);
    {
        std::scoped_lock lock(submitMutex);
        queue.submit(submitInfo, fence);
    }
    device.waitForFences(fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    device.destroyFence(fence);
}


vk::Fence Queue::submit(vk::CommandBuffer cmd)
{
    SAIGA_ASSERT(cmd, "invalid command buffer provided");

    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    auto fence                    = device.createFence({});
    {
        std::scoped_lock lock(submitMutex);
        queue.submit(submitInfo, fence);
    }
    return fence;
}

CommandPool Queue::createCommandPool(vk::CommandPoolCreateFlags commandPoolCreateFlags)
{
    std::scoped_lock pool_create_lock(commandPoolCreationMutex);
    CommandPool newCommandPool;
    newCommandPool.create(device, &commandPoolCreationMutex, queueFamilyIndex, commandPoolCreateFlags);
    VLOG(3) << "Creating command pool: " << static_cast<vk::CommandPool>(newCommandPool);
    commandPools.push_back(newCommandPool);
    return newCommandPool;
}

void Queue::submit(vk::SubmitInfo submitInfo, vk::Fence fence)
{
    std::scoped_lock submit_lock(submitMutex);
    queue.submit(submitInfo, fence);
}


}  // namespace Vulkan
}  // namespace Saiga
