//
// Created by Peter Eichinger on 2019-03-04.
//

#pragma once
#include "saiga/vulkan/svulkan.h"

namespace Saiga::Vulkan
{
class DescriptorSetLayout;
struct VulkanBase;

class DescriptorSet
{
   protected:
    VulkanBase* base;
    DescriptorSetLayout* layout;
    vk::DescriptorSet descriptorSet;
    DescriptorSet() : base(nullptr), layout(nullptr), descriptorSet(nullptr) {}
    DescriptorSet(VulkanBase* _base, DescriptorSetLayout* _layout, vk::DescriptorSet _descriptorSet)
        : base(_base), layout(_layout), descriptorSet(_descriptorSet)
    {
    }

   public:
    inline operator vk::ArrayProxy<const vk::DescriptorSet>() { return descriptorSet; }
    inline operator vk::DescriptorSet() { return descriptorSet; }
    inline virtual void update() = 0;
};
class StaticDescriptorSet final : public DescriptorSet
{
   public:
    StaticDescriptorSet() : DescriptorSet() {}
    StaticDescriptorSet(VulkanBase* base, DescriptorSetLayout* layout, vk::DescriptorSet descriptor)
        : DescriptorSet(base, layout, descriptor)
    {
    }

   public:
    inline void update() override {}
};

class DynamicDescriptorSet final : public DescriptorSet
{
   public:
    DynamicDescriptorSet() : DescriptorSet() {}
    DynamicDescriptorSet(VulkanBase* base, DescriptorSetLayout* layout, vk::DescriptorSet descriptor)
        : DescriptorSet(base, layout, descriptor)
    {
    }

   public:
    inline void update() override
    {
        // TODO: Modified check
    }
};
}  // namespace Saiga::Vulkan
