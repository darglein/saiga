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
#include "saiga/vulkan/CommandPool.h"
#include "VulkanTools.h"
#include "VulkanBuffer.hpp"

namespace Saiga{
namespace Vulkan{

struct SAIGA_GLOBAL VulkanBase
{
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceFeatures enabledFeatures;
    VkPhysicalDeviceMemoryProperties memoryProperties;
    std::vector<VkQueueFamilyProperties> queueFamilyProperties;
    std::vector<std::string> supportedExtensions;
    VkPipelineCache pipelineCache;

    /**
     * We store the transferQueue here so everyone can use it.
     * The graphics queues for rendering are created from the render engines.
     */
    Queue transferQueue;

    CommandPool commandPool;

    /** @brief Contains queue family indices */
    struct
    {
        uint32_t graphics = -1;
        uint32_t compute;
        uint32_t transfer;
        uint32_t present;
    } queueFamilyIndices;

    /**  @brief Typecast to VkDevice */
    operator VkDevice() { return device; }

    /**
        * Default constructor
        *
        * @param physicalDevice Physical device that is to be used
        */
     void bla(VkPhysicalDevice physicalDevice);

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
    uint32_t getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32 *memTypeFound = nullptr);

    /**
        * Get the index of a queue family that supports the requested queue flags
        *
        * @param queueFlags Queue flags to find a queue family index for
        *
        * @return Index of the queue family index that matches the flags
        *
        * @throw Throws an exception if no queue family index could be found that supports the requested flags
        */
    uint32_t getQueueFamilyIndex(VkQueueFlagBits queueFlags);
    uint32_t getPresentQueue(VkSurfaceKHR surface);

    /**
        * Create the logical device based on the assigned physical device, also gets default queue family indices
        *
        * @param enabledFeatures Can be used to enable certain features upon device creation
        * @param useSwapChain Set to false for headless rendering to omit the swapchain device extensions
        * @param requestedQueueTypes Bit flags specifying the queue types to be requested from the device
        *
        * @return VkResult of the device creation call
        */
    void createLogicalDevice(VkSurfaceKHR surface, VkPhysicalDeviceFeatures enabledFeatures, std::vector<const char*> enabledExtensions, bool useSwapChain = true, VkQueueFlags requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT| VK_QUEUE_TRANSFER_BIT);


    void init();
    /**
        * Create a buffer on the device
        *
        * @param usageFlags Usage flag bitmask for the buffer (i.e. index, vertex, uniform buffer)
        * @param memoryPropertyFlags Memory properties for this buffer (i.e. device local, host visible, coherent)
        * @param size Size of the buffer in byes
        * @param buffer Pointer to the buffer handle acquired by the function
        * @param memory Pointer to the memory handle acquired by the function
        * @param data Pointer to the data that should be copied to the buffer after creation (optional, if not set, no data is copied over)
        *
        * @return VK_SUCCESS if buffer handle and memory have been created and (optionally passed) data has been copied
        */
    VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, VkBuffer *buffer, VkDeviceMemory *memory, void *data = nullptr);

    /**
        * Create a buffer on the device
        *
        * @param usageFlags Usage flag bitmask for the buffer (i.e. index, vertex, uniform buffer)
        * @param memoryPropertyFlags Memory properties for this buffer (i.e. device local, host visible, coherent)
        * @param buffer Pointer to a vk::Vulkan buffer object
        * @param size Size of the buffer in byes
        * @param data Pointer to the data that should be copied to the buffer after creation (optional, if not set, no data is copied over)
        *
        * @return VK_SUCCESS if buffer handle and memory have been created and (optionally passed) data has been copied
        */
    VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, vks::Buffer *buffer, VkDeviceSize size, void *data = nullptr);

    /**
        * Copy buffer data from src to dst using VkCmdCopyBuffer
        *
        * @param src Pointer to the source buffer to copy from
        * @param dst Pointer to the destination buffer to copy tp
        * @param queue Pointer
        * @param copyRegion (Optional) Pointer to a copy region, if NULL, the whole buffer is copied
        *
        * @note Source and destionation pointers must have the approriate transfer usage flags set (TRANSFER_SRC / TRANSFER_DST)
        */
    void copyBuffer(vks::Buffer *src, vks::Buffer *dst, VkQueue queue, VkBufferCopy *copyRegion = nullptr);



    vk::CommandBuffer createAndBeginTransferCommand();

    /**
        * Finish command buffer recording and submit it to a queue
        *
        * @param commandBuffer Command buffer to flush
        * @param queue Queue to submit the command buffer to
        * @param free (Optional) Free the command buffer once it has been submitted (Defaults to true)
        *
        * @note The queue that the command buffer is submitted to must be from the same family index as the pool it was allocated from
        * @note Uses a fence to ensure command buffer has finished executing
        */
    void flushCommandBuffer2(VkCommandBuffer commandBuffer, VkQueue queue, bool free = true);

    /**
     * Submits the commandbuffer to the dedicated transfer queue and waits until it is finished.
     *
     */
    void transferAndWait(VkCommandBuffer commandBuffer, bool free = true);

    void endTransferWait(vk::CommandBuffer commandBuffer);

    /**
        * Check if an extension is supported by the (physical device)
        *
        * @param extension Name of the extension to check
        *
        * @return True if the extension is supported (present in the list read at device creation time)
        */
    bool extensionSupported(std::string extension);

};

}
}
