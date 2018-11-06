/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Queue.h"

namespace Saiga {
namespace Vulkan {

void Queue::create(vk::Device _device, uint32_t _queueFamilyIndex, uint32_t _queueIndex)
{
    device = _device;
    queueFamilyIndex = _queueFamilyIndex;
    queueIndex = _queueIndex;
    queue = _device.getQueue(_queueFamilyIndex,_queueIndex);
    SAIGA_ASSERT(queue);

    commandPool.create(device,_queueFamilyIndex);
}

void Queue::waitIdle()
{
    queue.waitIdle();
}

void Queue::destroy()
{
    waitIdle();
    commandPool.destroy();
}

void Queue::submitAndWait(vk::CommandBuffer cmd)
{
    if (!cmd)
    {
        return;
    }
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    auto fence = device.createFence({});
    queue.submit(submitInfo,fence);
    device.waitForFences(fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    device.destroyFence(fence);
}


vk::Fence Queue::submit(vk::CommandBuffer cmd) {
    SAIGA_ASSERT(cmd, "invalid command buffer provided");

    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    auto fence = device.createFence({});
    submitMutex.lock();
    queue.submit(submitInfo,fence);
    submitMutex.unlock();
    return fence;
}

CommandPool Queue::createCommandPool() {
    CommandPool newCommandPool;

    newCommandPool.create(device,queueFamilyIndex);

    return newCommandPool;
}



}
}
