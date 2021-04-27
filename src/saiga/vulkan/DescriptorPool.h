/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/svulkan.h"
#include <mutex>
namespace Saiga
{
namespace Vulkan
{
/**
 * Wrapper class for vk::DescriptorPool.
 */
class SAIGA_VULKAN_API DescriptorPool
{
   public:
    void create(vk::Device device, uint32_t maxSets, vk::ArrayProxy<const vk::DescriptorPoolSize> poolSizes,
                vk::DescriptorPoolCreateFlags flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);

    // Allocates a single descriptor set with the given layout
    vk::DescriptorSet allocateDescriptorSet(vk::DescriptorSetLayout layout);


    explicit operator vk::DescriptorPool() const { return descriptorPool; }

    explicit operator VkDescriptorPool() const { return descriptorPool; }

    void freeDescriptorSet(vk::DescriptorSet set);

    void destroy();

   private:
    std::mutex pool_mutex;
    vk::Device device;
    vk::DescriptorPool descriptorPool;
};


}  // namespace Vulkan
}  // namespace Saiga
