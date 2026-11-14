/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "DescriptorSet.h"

#include "DescriptorSetLayout.h"

namespace Saiga::Vulkan
{
void DescriptorSet::assign(uint32_t index, Buffer* buffer)
{
    SAIGA_ASSERT(descriptorSet && layout && buffer && buffer->memoryLocation());
    auto binding = layout->getBindingForIndex(index);
    auto time    = Clock::now();
    auto info    = buffer->createInfo();
    base->device.updateDescriptorSets(
        {vk::WriteDescriptorSet{descriptorSet, binding.binding, 0, 1, binding.descriptorType, nullptr, &info, nullptr}},
        nullptr);
    assigned_buffers[index] = std::make_pair(buffer, time);
    assigned_textures.erase(index);
}

void DescriptorSet::assign(uint32_t index, Texture* texture)
{
    SAIGA_ASSERT(descriptorSet && layout && texture && texture->memoryLocation);
    auto binding = layout->getBindingForIndex(index);
    auto time    = Clock::now();
    auto info    = texture->getDescriptorInfo();
    base->device.updateDescriptorSets(
        {vk::WriteDescriptorSet{descriptorSet, binding.binding, 0, 1, binding.descriptorType, &info, nullptr, nullptr}},
        nullptr);
    assigned_textures[index] = AssignedTexture{texture, nullptr, time};
    assigned_buffers.erase(index);
}

void DescriptorSet::assign(uint32_t index, Texture* texture, const Sampler& sampler)
{
    SAIGA_ASSERT(descriptorSet && layout && texture && texture->memoryLocation);
    auto binding = layout->getBindingForIndex(index);
    auto time    = Clock::now();
    auto info    = texture->getDescriptorInfo(sampler);
    base->device.updateDescriptorSets(
        {vk::WriteDescriptorSet{descriptorSet, binding.binding, 0, 1, binding.descriptorType, &info, nullptr, nullptr}},
        nullptr);
    assigned_textures[index] = AssignedTexture{texture, sampler, time};
    assigned_buffers.erase(index);
}

void DescriptorSet::release()
{
    // The pool is gone after VulkanBase::destroy().
    if (base && base->device && descriptorSet)
    {
        VLOG(3) << "~DescriptorSet(): " << descriptorSet;
        // Command buffers in flight may still use the set (VUID-vkFreeDescriptorSets-pDescriptorSets-00309).
        auto pool = &base->descriptorPool;
        auto set  = descriptorSet;
        base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
    }
    descriptorSet = nullptr;
}

DescriptorSet::~DescriptorSet()
{
    release();
}

DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept
    : base(other.base),
      layout(other.layout),
      descriptorSet(other.descriptorSet),
      assigned_textures(std::move(other.assigned_textures)),
      assigned_buffers(std::move(other.assigned_buffers))
{
    other.base          = nullptr;
    other.layout        = nullptr;
    other.descriptorSet = nullptr;
    other.assigned_textures.clear();
    other.assigned_buffers.clear();
}

DescriptorSet& DescriptorSet::operator=(DescriptorSet&& other) noexcept
{
    if (this != &other)
    {
        release();

        base              = other.base;
        layout            = other.layout;
        descriptorSet     = other.descriptorSet;
        assigned_textures = std::move(other.assigned_textures);
        assigned_buffers  = std::move(other.assigned_buffers);

        other.base          = nullptr;
        other.layout        = nullptr;
        other.descriptorSet = nullptr;
        other.assigned_textures.clear();
        other.assigned_buffers.clear();
    }

    return *this;
}

void DynamicDescriptorSet::update()
{
    if (!descriptorSet) return;

    vk::DescriptorSet new_set = nullptr;

    for (auto& entry : assigned_textures)
    {
        auto& texture = entry.second.texture;
        auto& time    = entry.second.time;
        if (!texture->memoryLocation) continue;

        if (texture->memoryLocation->modified_time > time)
        {
            if (!new_set)
            {
                // The current set may be used by frames in flight, so it is not updated in place.
                new_set = layout->createDuplicateSet(descriptorSet);
            }

            auto info = texture->getDescriptorInfo();
            if (entry.second.sampler) info.sampler = entry.second.sampler;
            base->device.updateDescriptorSets(layout->getWriteForBinding(entry.first, new_set, info), nullptr);

            time = texture->memoryLocation->modified_time;
        }
    }

    for (auto& entry : assigned_buffers)
    {
        auto& [buffer, time] = entry.second;
        auto* location       = buffer->memoryLocation();
        if (!location) continue;

        if (location->modified_time > time)
        {
            if (!new_set)
            {
                new_set = layout->createDuplicateSet(descriptorSet);
            }

            auto info = buffer->createInfo();
            base->device.updateDescriptorSets(layout->getWriteForBinding(entry.first, new_set, info), nullptr);

            time = location->modified_time;
        }
    }

    if (new_set)
    {
        // Free the old set once the frames that may use it have completed.
        release();
        descriptorSet = new_set;
    }
}
}  // namespace Saiga::Vulkan
