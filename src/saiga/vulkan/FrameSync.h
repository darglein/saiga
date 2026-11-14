/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/svulkan.h"

namespace Saiga
{
namespace Vulkan
{
/**
 * Synchronization objects of one frame in flight.
 *
 * The completion of a frame is tracked with the main queue's timeline semaphore (VulkanBase::frameTimeline), so no
 * fence is needed.
 */
class SAIGA_VULKAN_API FrameSync
{
   public:
    // Signaled by vkAcquireNextImageKHR, waited on by the frame's submit.
    vk::Semaphore imageAvailable = nullptr;

    // The frameTimeline value of the last frame that used this slot. 0 = never used.
    uint64_t timelineValue = 0;

    FrameSync() = default;
    ~FrameSync() { destroy(); }
    FrameSync(const FrameSync&)            = delete;
    FrameSync& operator=(const FrameSync&) = delete;

    void create(vk::Device device);
    void destroy();

   private:
    vk::Device device;
};

/**
 * The frame that is rendered, passed to VulkanRenderer::render.
 */
struct FrameInfo
{
    // Index of the frame in flight [0, framesInFlight). Use it for per-frame resources (command buffers, dynamic
    // buffers): when a frame index is reused, the previous frame with that index has completed.
    uint32_t frameIndex;

    // The acquired swap chain image.
    uint32_t imageIndex;

    // The submit must wait for imageAvailable (at color attachment output) ...
    vk::Semaphore imageAvailable;

    // ... signal renderComplete (waited on by the present) ...
    vk::Semaphore renderComplete;

    // ... and signal VulkanBase::frameTimeline with this value.
    uint64_t timelineValue;
};

}  // namespace Vulkan
}  // namespace Saiga
