/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "FrameSync.h"
#include "VulkanInitializers.hpp"
namespace Saiga {
namespace Vulkan {

void FrameSync::create(VkDevice device)
{
    // Wait fences to sync command buffer access
    VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    vkCreateFence(device, &fenceCreateInfo, nullptr, &frameFence);

    VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
    vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &presentComplete);
    vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderComplete);
}

void FrameSync::destroy(VkDevice device)
{
    vkDestroySemaphore(device, presentComplete, nullptr);
    vkDestroySemaphore(device, renderComplete, nullptr);
    vkDestroyFence(device, frameFence, nullptr);
}

void FrameSync::wait(VkDevice device)
{
    vkWaitForFences(device, 1, &frameFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    vkResetFences(device, 1, &frameFence);
}

}
}
