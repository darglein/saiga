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
#include <exception>

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
    Instance* instance;

   public:
    std::atomic_uint32_t current_frame;
    uint32_t numSwapchainFrames;
    VulkanParameters m_parameters;
    ~VulkanBase() { destroy(); }

    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::PhysicalDeviceFeatures enabledFeatures = {};

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
    Queue* transferQueue;

    inline bool has_dedicated_transfer() { return dedicated_transfer_queue != nullptr; }

    /**
     * A dedicated compute queue. If the GPU does not provide enough queues this will point to the same
     * queue as mainQueue. Depending on the GPU the queue may have other capabilities.
     * If the GPU provides dedicated Compute queues (without graphics capabilities) one of them will be used.
     */
    Queue* computeQueue;
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
     * @param requestedFeatures Can be used to enable certain features upon device creation
     * @param useSwapChain Set to false for headless rendering to omit the swapchain device extensions
     * @param requestedQueueTypes Bit flags specifying the queue types to be requested from the device
     *
     * @return VkResult of the device creation call
     */
    void createLogicalDevice(VulkanParameters& parameters, bool useSwapChain = true);


    [[deprecated("The functionality of this function was moved to createLogicalDevice(...)")]] void init(
        VulkanParameters params);

    void renderGUI();

    bool findQueueFamily(vk::QueueFlags flags, uint32_t& family, uint32_t offset = 0);

    bool findDedicatedQueueFamily(vk::QueueFlags flags, uint32_t& family);

    void initMemory(uint32_t swapchain_frames)
    {
        numSwapchainFrames = swapchain_frames;
        memory.init(this, swapchain_frames, m_parameters);
    }

    inline void finish_frame()
    {
        current_frame++;
        memory.update();
    }
};

}  // namespace Vulkan
}  // namespace Saiga
