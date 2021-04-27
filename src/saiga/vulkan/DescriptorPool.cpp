/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "DescriptorPool.h"

namespace Saiga
{
namespace Vulkan
{
void DescriptorPool::destroy()
{
    if (device && descriptorPool) device.destroyDescriptorPool(descriptorPool);
}

void DescriptorPool::create(vk::Device device, uint32_t maxSets, vk::ArrayProxy<const vk::DescriptorPoolSize> poolSizes,
                            vk::DescriptorPoolCreateFlags flags)
{
    this->device = device;
    vk::DescriptorPoolCreateInfo info(flags, maxSets, poolSizes.size(), poolSizes.data());

    descriptorPool = device.createDescriptorPool(info);
    SAIGA_ASSERT(descriptorPool);
}

vk::DescriptorSet DescriptorPool::allocateDescriptorSet(vk::DescriptorSetLayout layout)
{
    std::scoped_lock lock(pool_mutex);
    auto set = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo(descriptorPool, 1, &layout))[0];
    SAIGA_ASSERT(set);
    return set;
}

void DescriptorPool::freeDescriptorSet(vk::DescriptorSet set)
{
    std::scoped_lock lock(pool_mutex);
    device.freeDescriptorSets(descriptorPool, set);
}


}  // namespace Vulkan
}  // namespace Saiga
