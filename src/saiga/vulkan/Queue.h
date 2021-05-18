/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/export.h"
#include "saiga/vulkan/CommandPool.h"
#include "saiga/vulkan/svulkan.h"

#include <mutex>

namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API Queue
{
   public:
    // Create a primary commandpool for every queue.
    CommandPool commandPool;
    vk::Queue queue = nullptr;

    void create(vk::Device _device, uint32_t _queueFamilyIndex, uint32_t _queueIndex = 0,
                vk::CommandPoolCreateFlags commandPoolCreateFlags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    void waitIdle();
    void destroy();

    vk::Fence submit(vk::CommandBuffer cmd);

    void submit(vk::SubmitInfo submitInfo, vk::Fence fence);
    void submitAndWait(vk::CommandBuffer cmd);



    operator vk::Queue() const { return queue; }
    operator VkQueue() const { return queue; }

    uint32_t getQueueIndex() { return queueIndex; }
    uint32_t getQueueFamilyIndex() { return queueFamilyIndex; }

    CommandPool createCommandPool(
        vk::CommandPoolCreateFlags commandPoolCreateFlags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer);


    inline bool is_valid() { return queue; }

   private:
    std::vector<CommandPool> commandPools;
    std::mutex commandPoolCreationMutex;
    std::mutex submitMutex;
    uint32_t queueFamilyIndex;
    uint32_t queueIndex;
    vk::Device device;
};


}  // namespace Vulkan
}  // namespace Saiga
