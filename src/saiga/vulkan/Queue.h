/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/CommandPool.h"
#include <mutex>

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL Queue
{
public:
    // Create a primary commandpool for every queue.
    CommandPool commandPool;
    vk::Queue queue;

    void create(vk::Device _device, uint32_t _queueFamilyIndex, uint32_t _queueIndex = 0);
    void waitIdle();
    void destroy();

    vk::Fence submit(vk::CommandBuffer cmd);
    void submitAndWait(vk::CommandBuffer cmd);

    operator vk::Queue() const { return queue; }
    operator VkQueue() const { return queue; }

    uint32_t getQueueIndex() {return queueIndex;}
    uint32_t getQueueFamilyIndex() {return queueFamilyIndex;}

    CommandPool createCommandPool();
private:
    std::vector<CommandPool> commandPools;
    std::mutex commandPoolCreationMutex;
    std::mutex submitMutex;
    uint32_t queueFamilyIndex;
    uint32_t queueIndex;
    vk::Device device;

};


}
}
