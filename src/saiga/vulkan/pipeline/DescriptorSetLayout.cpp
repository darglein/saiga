//
// Created by Peter Eichinger on 2019-03-04.
//

#include "DescriptorSetLayout.h"

#include "saiga/core/util/assert.h"
#include "saiga/vulkan/Base.h"

namespace Saiga::Vulkan
{
DescriptorSetLayout::DescriptorSetLayout() : base(nullptr) {}

DescriptorSetLayout::DescriptorSetLayout(std::initializer_list<DescriptorSetLayout::BindingMapEntry> bindings)
    : base(nullptr)
{
    for (auto& entry : bindings)
    {
        addBinding(entry.first, entry.second);
    }
}

void DescriptorSetLayout::addBinding(uint32_t index, vk::DescriptorSetLayoutBinding binding)
{
    if (layout)
    {
        throw std::logic_error("If layout was created another binding cannot be added. Destroy it first.");
    }
    auto found = std::find_if(bindings.begin(), bindings.end(), [=](const auto& entry) {
        return entry.first == index || entry.second.binding == binding.binding;
    });

    if (found != bindings.end())
    {
        throw std::invalid_argument("Binding for this index or location already exists");
    }

    bindings.emplace(index, binding);
}

void DescriptorSetLayout::destroy()
{
    if (layout)
    {
        base->device.destroy(layout);
        layout = nullptr;
    }
}


vk::DescriptorSet DescriptorSetLayout::createRawDescriptorSet()
{
    if (!layout)
    {
        throw std::logic_error("Must call create() before creating descriptor sets.");
    }
    return base->descriptorPool.allocateDescriptorSet(layout);
}

StaticDescriptorSet DescriptorSetLayout::createDescriptorSet()
{
    if (!layout)
    {
        throw std::logic_error("Must call create() before creating descriptor sets.");
    }
    return StaticDescriptorSet(base, this, base->descriptorPool.allocateDescriptorSet(layout));
}

DynamicDescriptorSet DescriptorSetLayout::createDynamicDescriptorSet()
{
    if (!layout)
    {
        throw std::logic_error("Must call create() before creating descriptor sets.");
    }

    return DynamicDescriptorSet(base, this, base->descriptorPool.allocateDescriptorSet(layout));
}



void DescriptorSetLayout::create(VulkanBase* _base)
{
    base                 = _base;
    auto bindings_vector = std::vector<vk::DescriptorSetLayoutBinding>(bindings.size());

    std::transform(bindings.begin(), bindings.end(), bindings_vector.begin(),
                   [](const auto& entry) { return entry.second; });

    vk::DescriptorSetLayoutCreateInfo createInfo{vk::DescriptorSetLayoutCreateFlags(),
                                                 static_cast<uint32_t>(bindings_vector.size()), bindings_vector.data()};
    layout = base->device.createDescriptorSetLayout(createInfo);
}



}  // namespace Saiga::Vulkan
