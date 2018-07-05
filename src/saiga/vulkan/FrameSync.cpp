/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "FrameSync.h"
namespace Saiga {
namespace Vulkan {

void FrameSync::create(vk::Device device)
{
    vk::FenceCreateInfo fenceCreateInfo{
        vk::FenceCreateFlagBits::eSignaled
    };
    frameFence = device.createFence(fenceCreateInfo);

    vk::SemaphoreCreateInfo semaphoreCreateInfo {
        vk::SemaphoreCreateFlags()
    };
    device.createSemaphore(&semaphoreCreateInfo, nullptr, &imageVailable);
    device.createSemaphore(&semaphoreCreateInfo, nullptr, &renderComplete);
}

void FrameSync::destroy(vk::Device device)
{
//    vkDestroySemaphore(device, imageVailable, nullptr);
//    vkDestroySemaphore(device, renderComplete, nullptr);
//    vkDestroyFence(device, frameFence, nullptr);
    device.destroySemaphore(imageVailable);
    device.destroySemaphore(renderComplete);
    device.destroyFence(frameFence);
}

void FrameSync::wait(vk::Device device)
{
    device.waitForFences(frameFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    device.resetFences(frameFence);
}

}
}
