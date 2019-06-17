//
// Created by Peter Eichinger on 2019-03-04.
//

#include "DescriptorSet.h"

#include "DescriptorSetLayout.h"


void Saiga::Vulkan::DescriptorSet::assign(uint32_t index, Saiga::Vulkan::UniformBuffer* buffer)
{
    auto binding = layout->getBindingForIndex(index);
    auto info    = buffer->getDescriptorInfo();
    base->device.updateDescriptorSets(
        {vk::WriteDescriptorSet{descriptorSet, binding.binding, 0, 1, binding.descriptorType, nullptr, &info, nullptr}},
        nullptr);
}
void Saiga::Vulkan::DescriptorSet::assign(uint32_t index, Saiga::Vulkan::Texture* texture)
{
    auto binding = layout->getBindingForIndex(index);
    auto time    = Clock::now();
    auto info    = texture->getDescriptorInfo();
    base->device.updateDescriptorSets(
        {vk::WriteDescriptorSet{descriptorSet, binding.binding, 0, 1, binding.descriptorType, &info, nullptr, nullptr}},
        nullptr);
    assigned_textures[index] = std::make_pair(texture, time);
}
Saiga::Vulkan::DescriptorSet::~DescriptorSet()
{
    if (base && descriptorSet)
    {
        VLOG(3) << "~DescriptorSet(): " << descriptorSet;
        base->descriptorPool.freeDescriptorSet(descriptorSet);
        descriptorSet = nullptr;
    }
}
Saiga::Vulkan::DescriptorSet::DescriptorSet(Saiga::Vulkan::DescriptorSet&& other) noexcept
    : base(other.base),
      layout(other.layout),
      descriptorSet(other.descriptorSet),
      assigned_textures(std::move(other.assigned_textures))
{
    other.base          = nullptr;
    other.layout        = nullptr;
    other.descriptorSet = nullptr;
}
Saiga::Vulkan::DescriptorSet& Saiga::Vulkan::DescriptorSet::operator=(Saiga::Vulkan::DescriptorSet&& other) noexcept
{
    if (this != &other)
    {
        base              = other.base;
        layout            = other.layout;
        descriptorSet     = other.descriptorSet;
        assigned_textures = std::move(other.assigned_textures);

        other.base          = nullptr;
        other.layout        = nullptr;
        other.descriptorSet = nullptr;
    }

    return *this;
}
void Saiga::Vulkan::DynamicDescriptorSet::update()
{
    uint32_t current_frame = base->current_frame;

    if (m_old_set.first && m_old_set.second < current_frame)
    {
        base->descriptorPool.freeDescriptorSet(m_old_set.first);
        m_old_set = std::make_pair(nullptr, 0xFFFFFFF);
    }

    vk::DescriptorSet new_set = nullptr;
    std::vector<vk::DescriptorImageInfo> image_infos;
    std::vector<vk::WriteDescriptorSet> write_updates;

    image_infos.reserve(assigned_textures.size());
    write_updates.reserve(assigned_textures.size());
    for (auto& entry : assigned_textures)
    {
        auto& [texture, time] = entry.second;

        if (texture->memoryLocation->modified_time > time)
        {
            if (!new_set)
            {
                new_set   = layout->createDuplicateSet(descriptorSet);
                m_old_set = std::make_pair(descriptorSet, current_frame + base->numSwapchainFrames + 5);
            }

            base->device.updateDescriptorSets(layout->getWriteForBinding(entry.first, new_set,texture->getDescriptorInfo()),nullptr);

            time = texture->memoryLocation->modified_time;
        }
    }

    if (new_set)
    {
        descriptorSet = new_set;
    }
}

Saiga::Vulkan::DynamicDescriptorSet::~DynamicDescriptorSet()
{
    if (base && m_old_set.first)
    {
        base->descriptorPool.freeDescriptorSet(m_old_set.first);
        m_old_set = std::make_pair(nullptr, 0xFFFFFFFF);
    }
}
Saiga::Vulkan::DynamicDescriptorSet::DynamicDescriptorSet(Saiga::Vulkan::DynamicDescriptorSet&& other) noexcept
    : DescriptorSet(std::move(other)), m_old_set(std::move(other.m_old_set))
{
    other.m_old_set = std::make_pair(nullptr, 0xFFFFFFFF);
}
Saiga::Vulkan::DynamicDescriptorSet& Saiga::Vulkan::DynamicDescriptorSet::operator=(
    Saiga::Vulkan::DynamicDescriptorSet&& other) noexcept
{
    DescriptorSet::operator=(static_cast<DescriptorSet&&>(other));
    if (this != &other)
    {
        if (base && m_old_set.first)
        {
            base->descriptorPool.freeDescriptorSet(m_old_set.first);
        }
        m_old_set       = std::move(other.m_old_set);
        other.m_old_set = std::make_pair(nullptr, 0xFFFFFFFF);
    }

    return *this;
}
