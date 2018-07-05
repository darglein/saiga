/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/CommandPool.h"

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

    void submitAndWait(vk::CommandBuffer cmd);

    operator vk::Queue() const { return queue; }
    operator VkQueue() const { return queue; }
private:
    uint32_t queueFamilyIndex;
    uint32_t queueIndex;
    vk::Device device;

};


}
}
