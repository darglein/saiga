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
class SAIGA_VULKAN_API DescriptorSetLayout
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

    vk::DescriptorSet createRawDescriptorSet();
    StaticDescriptorSet createDescriptorSet();
    DynamicDescriptorSet createDynamicDescriptorSet();

    inline operator vk::DescriptorSetLayout() const
    {
        SAIGA_ASSERT(layout);
        return layout;
    }

    inline const vk::DescriptorSetLayoutBinding& getBindingForIndex(uint32_t index) const { return bindings.at(index); }

    vk::DescriptorSet createDuplicateSet(vk::DescriptorSet set)
    {
        auto duplicate = base->descriptorPool.allocateDescriptorSet(layout);

        std::vector<vk::CopyDescriptorSet> copies(bindings.size());

        std::transform(bindings.begin(), bindings.end(), copies.begin(),
                       [=](const std::pair<uint32_t, vk::DescriptorSetLayoutBinding>& binding) {
                           return vk::CopyDescriptorSet{
                               set, binding.second.binding, 0, duplicate, binding.second.binding, 0, 1};
                       });

        base->device.updateDescriptorSets(nullptr, copies);

        return duplicate;
    }

    inline vk::WriteDescriptorSet getWriteForBinding(uint32_t index, vk::DescriptorSet set,
                                                     const vk::DescriptorImageInfo& imageInfo) const
    {
        auto binding = getBindingForIndex(index);

        return {set, binding.binding, 0, 1, binding.descriptorType, &imageInfo};
    }
};

}  // namespace Saiga::Vulkan
