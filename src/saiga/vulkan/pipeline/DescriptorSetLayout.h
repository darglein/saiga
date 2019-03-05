//
// Created by Peter Eichinger on 2019-03-04.
//

#pragma once
#include "saiga/vulkan/svulkan.h"

#include "DescriptorSet.h"

#include <exception>
#include <map>

namespace Saiga::Vulkan
{
struct VulkanBase;
class DescriptorSetLayout
{
   public:
    using BindingMapType  = std::map<uint32_t, vk::DescriptorSetLayoutBinding>;
    using BindingMapEntry = BindingMapType::value_type;

   private:
    VulkanBase* base;
    BindingMapType bindings;
    vk::DescriptorSetLayout layout;

   public:
    void destroy();

    DescriptorSetLayout();
    DescriptorSetLayout(std::initializer_list<BindingMapEntry> bindings);
    void addBinding(uint32_t index, vk::DescriptorSetLayoutBinding binding);
    void create(VulkanBase* base);

    inline bool is_created() const { return static_cast<bool>(layout); }

    StaticDescriptorSet createDescriptorSet();
    DynamicDescriptorSet createDynamicDescriptorSet();

    inline operator vk::DescriptorSetLayout() const
    {
        SAIGA_ASSERT(layout);
        return layout;
    }
};

}  // namespace Saiga::Vulkan