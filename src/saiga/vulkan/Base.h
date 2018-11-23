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

#include <exception>
#include <assert.h>
#include <algorithm>

#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/Queue.h"
#include "saiga/vulkan/Parameters.h"
#include "saiga/vulkan/CommandPool.h"
#include "saiga/vulkan/DescriptorPool.h"
#include "VulkanTools.h"
#include "VulkanBuffer.hpp"
#include "saiga/vulkan/memory/VulkanMemory.h"

namespace Saiga{
namespace Vulkan{

struct SAIGA_GLOBAL VulkanBase
{
    vk::PhysicalDevice physicalDevice;
    vk::PhysicalDeviceFeatures enabledFeatures = {};
    vk::Device device;

    Saiga::Vulkan::Memory::VulkanMemory memory;
    vk::PhysicalDeviceMemoryProperties memoryProperties;

    std::vector<vk::QueueFamilyProperties> queueFamilyProperties;

    vk::PipelineCache pipelineCache;

    /**
     * We store the transferQueue here so everyone can use it.
     * The graphics queues for rendering are created from the render engines.
     */
    Queue transferQueue;

    bool secondaryQueueAvailable = false;
    Queue secondaryTransferQueue;

    // A commandpool for transfer commands that are sync-submitted. (this is not the command pool used for rendering)
    CommandPool commandPool;

    // A large descriptor pool which should be used by the application
    // The size is controlled by the vulkan parameters
    DescriptorPool descriptorPool;




    /** @brief Contains queue family indices */
    struct
    {
        uint32_t graphics = -1;
        uint32_t compute;
        uint32_t transfer;
        uint32_t present;
    } queueFamilyIndices;

    operator vk::Device() { return device; }


    void setPhysicalDevice(vk::PhysicalDevice physicalDevice);

    /**
        * Default destructor
        *
        * @note Frees the logical device
        */
    void destroy();

    /**
        * Get the index of a memory type that has all the requested property bits set
        *
        * @param typeBits Bitmask with bits set for each memory type supported by the resource to request for (from VkMemoryRequirements)
        * @param properties Bitmask of properties for the memory type to request
        * @param (Optional) memTypeFound Pointer to a bool that is set to true if a matching memory type has been found
        *
        * @return Index of the requested memory type
        *
        * @throw Throws an exception if memTypeFound is null and no memory type could be found that supports the requested properties
        */
    uint32_t getMemoryType(uint32_t typeBits, vk::MemoryPropertyFlags properties, VkBool32 *memTypeFound = nullptr);

    void printAvailableMemoryTypes();

    /**
        * Get the index of a queue family that supports the requested queue flags
        *
        * @param queueFlags Queue flags to find a queue family index for
        *
        * @return Index of the queue family index that matches the flags
        *
        * @throw Throws an exception if no queue family index could be found that supports the requested flags
        */
    uint32_t getQueueFamilyIndex(vk::QueueFlags queueFlags);
    uint32_t getPresentQueue(vk::SurfaceKHR surface);

    /**
        * Create the logical device based on the assigned physical device, also gets default queue family indices
        *
        * @param requestedFeatures Can be used to enable certain features upon device creation
        * @param useSwapChain Set to false for headless rendering to omit the swapchain device extensions
        * @param requestedQueueTypes Bit flags specifying the queue types to be requested from the device
        *
        * @return VkResult of the device creation call
        */
    void createLogicalDevice(
            vk::SurfaceKHR surface,
            vk::PhysicalDeviceFeatures requestedFeatures,
            std::vector<const char*> enabledExtensions,
            bool useSwapChain = true,
            vk::QueueFlags requestedQueueTypes =  vk::QueueFlagBits::eGraphics |  vk::QueueFlagBits::eCompute |  vk::QueueFlagBits::eTransfer,
            bool createSecondaryTransferQueue = false);


    void init( VulkanParameters params );

    vk::CommandBuffer createAndBeginTransferCommand();

    /**
     * Submits the command buffer to the queue and waits until it is completed with a fence.
     */
    void submitAndWait(vk::CommandBuffer commandBuffer, vk::Queue queue);

    void endTransferWait(vk::CommandBuffer commandBuffer);

    void printAvailableQueueFamilies();
};

}
}
