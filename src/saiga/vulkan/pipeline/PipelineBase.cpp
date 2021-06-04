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
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    for (auto& l : descriptorSetLayouts) l.destroy();
    device = nullptr;
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



void PipelineBase::createPipelineLayout()
{
    SAIGA_ASSERT(isInitialized());

    std::vector<vk::DescriptorSetLayout> layouts(descriptorSetLayouts.size());

    std::transform(descriptorSetLayouts.begin(), descriptorSetLayouts.end(), layouts.begin(),
                   [](auto& entry) { return static_cast<vk::DescriptorSetLayout>(entry); });
    vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), layouts.size(),
                                                           layouts.data(), pushConstantRanges.size(),
                                                           pushConstantRanges.data());
    pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
    SAIGA_ASSERT(pipelineLayout);
}


}  // namespace Vulkan
}  // namespace Saiga
