/*
 * Vulkan device class
 *
 * Encapsulates a physical Vulkan device and it's logical representation
 *
 * Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include "saiga/vulkan/CommandPool.h"
#include "saiga/vulkan/DescriptorPool.h"
#include "saiga/vulkan/Parameters.h"
#include "saiga/vulkan/Queue.h"
#include "saiga/vulkan/memory/VulkanMemory.h"
#include "saiga/vulkan/svulkan.h"

#include "VulkanBuffer.hpp"
#include "VulkanTools.h"

#include <algorithm>
#include <assert.h>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>

namespace Saiga
{
namespace Vulkan
{
class Instance;

struct SAIGA_VULKAN_API VulkanBase
{
   private:
    // TODO: Rename to backing fields for dedicated queues
    std::unique_ptr<Queue> dedicated_compute_queue;
    std::unique_ptr<Queue> dedicated_transfer_queue;
    Instance* instance = nullptr;

    // Deferred deletions, see deferDelete().
    std::mutex deletionMutex;
    std::deque<std::pair<uint64_t, std::function<void()>>> deletionQueue;

   public:
    /**
     * The number of the frame that is currently recorded. Incremented by finish_frame() after the frame was submitted.
     * Frame f signals frameTimeline with the value f + 1 when all its commands on the main queue have completed.
     */
    std::atomic_uint32_t current_frame = 0;
    uint32_t numSwapchainFrames        = 0;
    VulkanParameters m_parameters;

    // Timeline semaphore of the main queue, see current_frame.
    vk::Semaphore frameTimeline;

    // The value frameTimeline is signaled with when the current frame is complete.
    uint64_t currentFrameSignalValue() const { return uint64_t(current_frame) + 1; }

    // The value frameTimeline has reached: all frames f with f + 1 <= value are complete.
    uint64_t completedFrameValue() const;

    // Blocks until frameTimeline reaches the value.
    void waitFrameValue(uint64_t value);

    /**
     * Executes the function (e.g. destroying a Vulkan object) once the GPU has finished all frames that were submitted
     * so far, including the frame that is currently recorded. Use this for objects that may still be referenced by
     * command buffers in flight.
     */
    void deferDelete(std::function<void()> f);

    // Runs the deferred deletions whose frames are complete. Called every frame by finish_frame().
    void processDeferredDeletes();

    // Waits for the device and runs all deferred deletions.
    void flushDeferredDeletes();
    ~VulkanBase();

    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::PhysicalDeviceFeatures enabledFeatures = {};
    // The API version the device is used with (SAIGA_VULKAN_API_VERSION = 1.3)
    uint32_t apiVersion = 0;

    Saiga::Vulkan::Memory::VulkanMemory memory;

    std::vector<vk::QueueFamilyProperties> queueFamilyProperties;

    vk::PipelineCache pipelineCache;

    /**
     * The main queue must be able to do Graphics and Compute (and Transfer).
     */
    Queue mainQueue;

    /**
     * This queue is a dedicated transfer queue. If the GPU does not provide enough queues this will point to the same
     * queue as mainQueue. Depending on the GPU the queue may have other capabilities. If the GPU provides dedicated
     * transfer queues (without graphics and compute capabilities) one of them will be used.
     */
    Queue* transferQueue = nullptr;

    inline bool has_dedicated_transfer() { return dedicated_transfer_queue != nullptr; }

    /**
     * A dedicated compute queue. If the GPU does not provide enough queues this will point to the same
     * queue as mainQueue. Depending on the GPU the queue may have other capabilities.
     * If the GPU provides dedicated Compute queues (without graphics capabilities) one of them will be used.
     */
    Queue* computeQueue = nullptr;
    inline bool has_dedicated_compute() { return dedicated_compute_queue != nullptr; }


    // A large descriptor pool which should be used by the application
    // The size is controlled by the vulkan parameters
    DescriptorPool descriptorPool;

    VulkanBase();
    operator vk::Device() { return device; }


    void setPhysicalDevice(Instance& instance, vk::PhysicalDevice physicalDevice);

    /**
     * Default destructor
     *
     * @note Frees the logical device
     */
    void destroy();

    /**
     * Create the logical device based on the assigned physical device, also gets default queue family indices
     *
     * Enables the Vulkan 1.3 features saiga uses (dynamic rendering, synchronization2, timeline semaphores) and the
     * requested core features that are available.
     *
     * @param surface If set, the swapchain extension is enabled and the main queue family must support presenting to
     *                this surface. Pass nullptr for headless rendering.
     */
    void createLogicalDevice(VulkanParameters& parameters, vk::SurfaceKHR surface);

    void renderGUI();

    bool findQueueFamily(vk::QueueFlags flags, uint32_t& family, uint32_t offset = 0);

    bool findDedicatedQueueFamily(vk::QueueFlags flags, uint32_t& family);

    void initMemory(uint32_t swapchain_frames)
    {
        numSwapchainFrames = swapchain_frames;
        memory.init(this, swapchain_frames, m_parameters);
    }

    // Called after the current frame was submitted (with frameTimeline = currentFrameSignalValue()).
    inline void finish_frame()
    {
        current_frame++;
        processDeferredDeletes();
        memory.update();
    }
};

}  // namespace Vulkan
}  // namespace Saiga
