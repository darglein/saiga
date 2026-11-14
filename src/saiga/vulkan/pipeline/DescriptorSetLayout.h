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

/**
 * The bindings of a descriptor set layout and (after create()) the vk::DescriptorSetLayout handle.
 *
 * The handle is not owned in the RAII sense: there is no destructor, destroy() must be called explicitly and is
 * idempotent. A copy shares the handle, so destroy() must be called on exactly one of the copies (PipelineBase copies
 * the layouts into a deferred deletion and destroys them there).
 */
class SAIGA_VULKAN_API DescriptorSetLayout
{
   public:
    using BindingMapType  = std::map<uint32_t, vk::DescriptorSetLayoutBinding>;
    using BindingMapEntry = BindingMapType::value_type;

   private:
    VulkanBase* base = nullptr;
    BindingMapType bindings;
    vk::DescriptorSetLayout layout = nullptr;

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

    // A new set with the same descriptors as set.
    vk::DescriptorSet createDuplicateSet(vk::DescriptorSet set)
    {
        auto duplicate = base->descriptorPool.allocateDescriptorSet(layout);

        std::vector<vk::CopyDescriptorSet> copies(bindings.size());

        std::transform(bindings.begin(), bindings.end(), copies.begin(),
                       [=](const std::pair<uint32_t, vk::DescriptorSetLayoutBinding>& binding)
                       {
                           return vk::CopyDescriptorSet{
                               set, binding.second.binding,        0, duplicate, binding.second.binding,
                               0,   binding.second.descriptorCount};
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
