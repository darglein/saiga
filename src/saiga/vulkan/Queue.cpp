/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Queue.h"
#include "saiga/util/easylogging++.h"

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

    LOG(INFO) << "Destroying queue with " << commandPools.size() << " command pools" ;
    for(auto& pool : commandPools) {
        LOG(INFO) << "Destroying command pool: "<<static_cast<vk::CommandPool>(pool);
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
    commandPoolCreationMutex.lock();

    CommandPool newCommandPool;
    newCommandPool.create(device,queueFamilyIndex);
    LOG(INFO) << "Creating command pool: " << static_cast<vk::CommandPool>(newCommandPool);
//    LOG(INFO) << commandPools.size();
    commandPools.push_back(newCommandPool);
//    LOG(INFO) << commandPools.size();
    commandPoolCreationMutex.unlock();
    return newCommandPool;
}



}
}
