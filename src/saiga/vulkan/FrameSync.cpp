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
void FrameSync::create(vk::Device _device)
{
    SAIGA_ASSERT(!imageAvailable);
    device         = _device;
    imageAvailable = device.createSemaphore(vk::SemaphoreCreateInfo{});
    timelineValue  = 0;
}

void FrameSync::destroy()
{
    if (imageAvailable)
    {
        device.destroySemaphore(imageAvailable);
        imageAvailable = nullptr;
    }
}

}  // namespace Vulkan
}  // namespace Saiga
