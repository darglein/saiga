/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "FrameSync.h"
namespace Saiga
{
namespace Vulkan
{
void FrameSync::create(vk::Device device)
{
    this->device = device;

    SAIGA_ASSERT(!imageAvailable);

    vk::FenceCreateInfo fenceCreateInfo{vk::FenceCreateFlagBits::eSignaled};
    frameFence = device.createFence(fenceCreateInfo);

    vk::SemaphoreCreateInfo semaphoreCreateInfo{vk::SemaphoreCreateFlags()};
    device.createSemaphore(&semaphoreCreateInfo, nullptr, &imageAvailable);
    device.createSemaphore(&semaphoreCreateInfo, nullptr, &renderComplete);
    defragMayStart = device.createSemaphore(semaphoreCreateInfo);
}

void FrameSync::destroy()
{
    if (imageAvailable)
    {
        device.destroySemaphore(imageAvailable);
        device.destroySemaphore(renderComplete);
        device.destroy(defragMayStart);
        device.destroyFence(frameFence);
        imageAvailable = nullptr;
    }
}

void FrameSync::wait()
{
    device.waitForFences(frameFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    device.resetFences(frameFence);
}

}  // namespace Vulkan
}  // namespace Saiga
