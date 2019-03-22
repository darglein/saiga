/**
 * Copyright (c) 2017 Darius Rückert
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
    commandPool.create(device, _queueFamilyIndex, commandPoolCreateFlags);
}

void Queue::waitIdle()
{
    LOG(INFO) << "Queue: wait idle...";
    submitMutex.lock();
    queue.waitIdle();
    submitMutex.unlock();
    LOG(INFO) << "Queue: wait idle done.";
}

void Queue::destroy()
{
    waitIdle();

    LOG(INFO) << "Destroying queue with " << commandPools.size() << " command pools";
    for (auto& pool : commandPools)
    {
        LOG(INFO) << "Destroying command pool: " << static_cast<vk::CommandPool>(pool);
        pool.destroy();
    }

    commandPool.destroy();

    //    device.destroy(*this);
}

void Queue::submitAndWait(vk::CommandBuffer cmd)
{
    if (!cmd)
    {
        return;
    }
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    vk::FenceCreateInfo fci{};
    submitMutex.lock();
    auto fence = device.createFence(fci);
    queue.submit(submitInfo, fence);
    submitMutex.unlock();
    device.waitForFences(fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    device.destroyFence(fence);
}


vk::Fence Queue::submit(vk::CommandBuffer cmd)
{
    SAIGA_ASSERT(cmd, "invalid command buffer provided");

    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    submitMutex.lock();
    auto fence = device.createFence({});
    queue.submit(submitInfo, fence);
    submitMutex.unlock();
    return fence;
}

CommandPool Queue::createCommandPool(vk::CommandPoolCreateFlags commandPoolCreateFlags)
{
    commandPoolCreationMutex.lock();

    CommandPool newCommandPool;
    newCommandPool.create(device, queueFamilyIndex, commandPoolCreateFlags);
    LOG(INFO) << "Creating command pool: " << static_cast<vk::CommandPool>(newCommandPool);
    commandPools.push_back(newCommandPool);
    commandPoolCreationMutex.unlock();
    return newCommandPool;
}

void Queue::submit(vk::SubmitInfo submitInfo, vk::Fence fence)
{
    submitMutex.lock();
    queue.submit(submitInfo, fence);
    submitMutex.unlock();
}


}  // namespace Vulkan
}  // namespace Saiga
