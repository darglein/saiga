/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "PipelineBase.h"
#include "saiga/vulkan/VulkanInitializers.hpp"
#include "saiga/vulkan/Vertex.h"

namespace Saiga {
namespace Vulkan {


PipelineBase::PipelineBase(vk::PipelineBindPoint type)
    : type(type)
{

}

void PipelineBase::init(VulkanBase &base, uint32_t numDescriptorSetLayouts)
{
    this->base = &base;
    device = base.device;
    descriptorSetLayouts.resize(numDescriptorSetLayouts);
}

void PipelineBase::destroy()
{
    if(!device)
        return;
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    for(auto& l :descriptorSetLayouts)
        vkDestroyDescriptorSetLayout(device, l, nullptr);
}

vk::DescriptorSet PipelineBase::createDescriptorSet(uint32_t id)
{
    SAIGA_ASSERT(isInitialized());
    SAIGA_ASSERT(id >= 0 && id < descriptorSetLayouts.size());
    return base->descriptorPool.allocateDescriptorSet(descriptorSetLayouts[id]);
}

void PipelineBase::bind(vk::CommandBuffer cmd)
{
    cmd.bindPipeline(type,pipeline);
}

void PipelineBase::bindDescriptorSets(vk::CommandBuffer cmd, vk::ArrayProxy<const vk::DescriptorSet> descriptorSets, uint32_t firstSet, vk::ArrayProxy<const uint32_t> dynamicOffsets)
{
    cmd.bindDescriptorSets(type,pipelineLayout,firstSet,descriptorSets,dynamicOffsets);
}

void PipelineBase::pushConstant(vk::CommandBuffer cmd, vk::ShaderStageFlags stage, size_t size, const void *data, size_t offset)
{
    cmd.pushConstants(pipelineLayout,stage,offset,size,data)   ;
}

void PipelineBase::addDescriptorSetLayout(std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings, uint32_t id)
{
    SAIGA_ASSERT(isInitialized());
    SAIGA_ASSERT(id >= 0 && id < descriptorSetLayouts.size());
    vk::DescriptorSetLayoutCreateInfo descriptorLayout(
                vk::DescriptorSetLayoutCreateFlags(),
                setLayoutBindings.size(),
                setLayoutBindings.data()
                );
    auto setLayout = device.createDescriptorSetLayout(descriptorLayout);
    SAIGA_ASSERT(setLayout);
    descriptorSetLayouts[id] = setLayout;
}

void PipelineBase::addPushConstantRange(vk::PushConstantRange pcr)
{
    SAIGA_ASSERT(isInitialized());
    pushConstantRanges.push_back(pcr);
}

void PipelineBase::createPipelineLayout( )
{
    SAIGA_ASSERT(isInitialized());
    vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo(
                vk::PipelineLayoutCreateFlags(),
                descriptorSetLayouts.size(),
                descriptorSetLayouts.data(),
                pushConstantRanges.size(),
                pushConstantRanges.data()
                );
    pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
    SAIGA_ASSERT(pipelineLayout);
}




}
}
