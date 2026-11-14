//
// Created by Peter Eichinger on 2019-03-04.
//

#include "DescriptorSet.h"

#include "DescriptorSetLayout.h"

namespace Saiga::Vulkan
{
void DescriptorSet::assign(uint32_t index, UniformBuffer* buffer)
{
    SAIGA_ASSERT(descriptorSet && layout && buffer);
    auto binding = layout->getBindingForIndex(index);
    auto info    = buffer->getDescriptorInfo();
    base->device.updateDescriptorSets(
        {vk::WriteDescriptorSet{descriptorSet, binding.binding, 0, 1, binding.descriptorType, nullptr, &info, nullptr}},
        nullptr);
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
    assigned_textures[index] = std::make_pair(texture, time);
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
      assigned_textures(std::move(other.assigned_textures))
{
    other.base          = nullptr;
    other.layout        = nullptr;
    other.descriptorSet = nullptr;
    other.assigned_textures.clear();
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

        other.base          = nullptr;
        other.layout        = nullptr;
        other.descriptorSet = nullptr;
        other.assigned_textures.clear();
    }

    return *this;
}

void DynamicDescriptorSet::update()
{
    if (!descriptorSet) return;

    vk::DescriptorSet new_set = nullptr;

    for (auto& entry : assigned_textures)
    {
        auto& [texture, time] = entry.second;
        if (!texture->memoryLocation) continue;

        if (texture->memoryLocation->modified_time > time)
        {
            if (!new_set)
            {
                // The current set may be used by frames in flight, so it is not updated in place.
                new_set = layout->createDuplicateSet(descriptorSet);
            }

            auto info = texture->getDescriptorInfo();
            base->device.updateDescriptorSets(layout->getWriteForBinding(entry.first, new_set, info), nullptr);

            time = texture->memoryLocation->modified_time;
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
