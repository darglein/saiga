/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/vulkanBase.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL CommandPool
{
public:
    vk::CommandPool cmd_pool;

    CommandPool();
    ~CommandPool();

    void create(VulkanBase &base, uint32_t queueFamilyIndex, vk::CommandPoolCreateFlags flags = vk::CommandPoolCreateFlags());

    vk::CommandBuffer createCommandBuffer(VulkanBase &base,  vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
};

}
}
