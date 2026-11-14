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

    bool moved = false;
    for (auto& entry : assigned_textures)
    {
        auto& location = entry.second.texture->memoryLocation;
        if (location && location->modifiedTime() > entry.second.time) moved = true;
    }
    for (auto& entry : assigned_buffers)
    {
        auto* location = entry.second.first->memoryLocation();
        if (location && location->modifiedTime() > entry.second.second) moved = true;
    }
    if (!moved) return;

    // The current set may be used by frames in flight, so it is not updated in place. The assigned resources are
    // written again instead of copied: the old set still refers to the image, view, sampler or buffer a moved
    // resource had before, which may be destroyed already.
    std::set<uint32_t> assigned;
    for (auto& entry : assigned_textures) assigned.insert(entry.first);
    for (auto& entry : assigned_buffers) assigned.insert(entry.first);
    vk::DescriptorSet new_set = layout->createDuplicateSet(descriptorSet, assigned);

    for (auto& entry : assigned_textures)
    {
        auto& texture = entry.second.texture;
        if (!texture->memoryLocation) continue;

        // The time before the handles are read: a move in between is detected by the next update.
        entry.second.time = texture->memoryLocation->modifiedTime();
        auto info         = texture->getDescriptorInfo();
        if (entry.second.sampler) info.sampler = entry.second.sampler;
        base->device.updateDescriptorSets(layout->getWriteForBinding(entry.first, new_set, info), nullptr);
    }

    for (auto& entry : assigned_buffers)
    {
        auto& [buffer, time] = entry.second;
        auto* location       = buffer->memoryLocation();
        if (!location) continue;

        time      = location->modifiedTime();
        auto info = buffer->createInfo();
        base->device.updateDescriptorSets(layout->getWriteForBinding(entry.first, new_set, info), nullptr);
    }

    // Free the old set once the frames that may use it have completed.
    release();
    descriptorSet = new_set;
}
}  // namespace Saiga::Vulkan
