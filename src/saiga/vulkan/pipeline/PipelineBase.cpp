/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "PipelineBase.h"

#include "saiga/vulkan/Vertex.h"
#include "saiga/vulkan/VulkanInitializers.hpp"

namespace Saiga
{
namespace Vulkan
{
PipelineBase::PipelineBase(vk::PipelineBindPoint type) : type(type) {}

void PipelineBase::init(VulkanBase& base, uint32_t numDescriptorSetLayouts)
{
    this->base = &base;
    device     = base.device;
    descriptorSetLayouts.resize(numDescriptorSetLayouts);
}

void PipelineBase::destroy()
{
    if (!device) return;
    VLOG(3) << "Destroying pipeline " << pipeline;

    // Command buffers in flight may still use the pipeline and its layouts. The lambda gets copies of the layouts
    // (they share the handle, only the copies are destroyed): the descriptor sets created from them point to the
    // elements of descriptorSetLayouts, which stay valid as long as this object.
    replacePipeline(nullptr);
    auto dev     = device;
    auto layout  = pipelineLayout;
    auto layouts = descriptorSetLayouts;
    base->deferDelete(
        [dev, layout, layouts]() mutable
        {
            if (layout) dev.destroyPipelineLayout(layout);
            for (auto& l : layouts) l.destroy();
        });

    pipeline       = nullptr;
    pipelineLayout = nullptr;
    pushConstantRanges.clear();
    device = nullptr;
}

void PipelineBase::replacePipeline(vk::Pipeline newPipeline)
{
    if (pipeline)
    {
        auto dev         = device;
        auto old         = pipeline;
        auto oldLayout   = pooledLayout;
        auto key         = pooledKey;
        VulkanBase* pool = base;
        base->deferDelete(
            [dev, old, oldLayout, key, pool]()
            {
                if (oldLayout)
                    pool->returnPooledPipeline(key, old, oldLayout);
                else
                    dev.destroyPipeline(old);
            });
    }
    pipeline     = newPipeline;
    pooledLayout = nullptr;
    pooledKey.clear();
}

void PipelineBase::setPipeline(const std::string& key, const std::function<vk::Pipeline(vk::PipelineLayout)>& make)
{
    if (key.empty())
    {
        replacePipeline(make(pipelineLayout));
        return;
    }
    vk::Pipeline p;
    vk::PipelineLayout l;
    if (!base->takePooledPipeline(key, p, l))
    {
        l = newPipelineLayout();
        p = make(l);
    }
    replacePipeline(p);
    pooledLayout = l;
    pooledKey    = key;
}

void PipelineBase::writeLayoutKey(std::string& key) const
{
    auto raw = [&](uint64_t v) { key.append(reinterpret_cast<const char*>(&v), sizeof(v)); };
    raw(descriptorSetLayouts.size());
    for (const DescriptorSetLayout& l : descriptorSetLayouts)
    {
        raw(l.getBindings().size());
        for (const auto& [index, b] : l.getBindings())
        {
            raw(index);
            raw(b.binding);
            raw(uint64_t(b.descriptorType));
            raw(b.descriptorCount);
            raw(uint64_t(VkShaderStageFlags(b.stageFlags)));
            // Immutable samplers by handle (another scene's samplers: no reuse)
            raw(b.pImmutableSamplers ? 1 : 0);
            if (b.pImmutableSamplers)
                for (uint32_t i = 0; i < b.descriptorCount; ++i)
                    raw(uint64_t(reinterpret_cast<uintptr_t>(static_cast<VkSampler>(b.pImmutableSamplers[i]))));
        }
    }
    raw(pushConstantRanges.size());
    for (const auto& r : pushConstantRanges)
    {
        raw(uint64_t(VkShaderStageFlags(r.stageFlags)));
        raw(r.offset);
        raw(r.size);
    }
}

vk::DescriptorSet PipelineBase::createRawDescriptorSet(uint32_t id)
{
    SAIGA_ASSERT(isInitialized());
    SAIGA_ASSERT(id >= 0 && id < descriptorSetLayouts.size());
    return descriptorSetLayouts[id].createRawDescriptorSet();
}

StaticDescriptorSet PipelineBase::createDescriptorSet(uint32_t id)
{
    SAIGA_ASSERT(isInitialized());
    SAIGA_ASSERT(id >= 0 && id < descriptorSetLayouts.size());
    return descriptorSetLayouts[id].createDescriptorSet();
}

DynamicDescriptorSet PipelineBase::createDynamicDescriptorSet(uint32_t id)
{
    SAIGA_ASSERT(isInitialized());
    SAIGA_ASSERT(id >= 0 && id < descriptorSetLayouts.size());
    return descriptorSetLayouts[id].createDynamicDescriptorSet();
}

bool PipelineBase::bind(vk::CommandBuffer cmd)
{
    if (checkShader())
    {
        cmd.bindPipeline(type, pipeline);
        return true;
    }
    else
    {
        return false;
    }
}



void PipelineBase::pushConstant(vk::CommandBuffer cmd, vk::ShaderStageFlags stage, size_t size, const void* data,
                                size_t offset)
{
    cmd.pushConstants(pipelineLayout, stage, offset, size, data);
}

void PipelineBase::addDescriptorSetLayout(const DescriptorSetLayout& layout, uint32_t id)
{
    SAIGA_ASSERT(isInitialized());
    SAIGA_ASSERT(id >= 0 && id < descriptorSetLayouts.size());

    SAIGA_ASSERT(!layout.is_created(), "Creation must not be done beforehand");

    descriptorSetLayouts[id] = layout;

    descriptorSetLayouts[id].create(base);
}

void PipelineBase::addPushConstantRange(vk::PushConstantRange pcr)
{
    SAIGA_ASSERT(isInitialized());
    pushConstantRanges.push_back(pcr);
}



vk::PipelineLayout PipelineBase::newPipelineLayout() const
{
    std::vector<vk::DescriptorSetLayout> layouts(descriptorSetLayouts.size());

    std::transform(descriptorSetLayouts.begin(), descriptorSetLayouts.end(), layouts.begin(),
                   [](auto& entry) { return static_cast<vk::DescriptorSetLayout>(entry); });
    vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), layouts.size(),
                                                           layouts.data(), pushConstantRanges.size(),
                                                           pushConstantRanges.data());
    vk::PipelineLayout l = device.createPipelineLayout(pPipelineLayoutCreateInfo);
    SAIGA_ASSERT(l);
    return l;
}

void PipelineBase::createPipelineLayout()
{
    SAIGA_ASSERT(isInitialized());
    if (pipelineLayout)
    {
        // create() called again
        auto dev = device;
        auto old = pipelineLayout;
        base->deferDelete([dev, old]() { dev.destroyPipelineLayout(old); });
    }
    pipelineLayout = newPipelineLayout();
}


}  // namespace Vulkan
}  // namespace Saiga
