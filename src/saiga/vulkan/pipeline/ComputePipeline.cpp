/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ComputePipeline.h"
#include "saiga/vulkan/VulkanInitializers.hpp"
#include "saiga/vulkan/Vertex.h"

namespace Saiga {
namespace Vulkan {

void ComputePipelineInfo::setShader(Saiga::Vulkan::ShaderModule& shader)
{
    shaderStage = shader.createPipelineInfo();
}

vk::ComputePipelineCreateInfo ComputePipelineInfo::createCreateInfo(vk::PipelineLayout pipelineLayout)
{
    vk::ComputePipelineCreateInfo pipelineCreateInfo(
                vk::PipelineCreateFlags(),
                shaderStage,
                pipelineLayout,
                vk::Pipeline(),
                0
                );
    return pipelineCreateInfo;
}



void ComputePipeline::destroy()
{
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    for(auto& l :descriptorSetLayout)
        vkDestroyDescriptorSetLayout(device, l, nullptr);
    vkDestroyDescriptorPool(device,descriptorPool,nullptr);
    shader.destroy(device);

}




void ComputePipeline::createDescriptorSetLayout(std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings)
{
    SAIGA_ASSERT(device);
    //    VkDescriptorSetLayoutCreateInfo descriptorLayout =
    //            vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    vk::DescriptorSetLayoutCreateInfo descriptorLayout(vk::DescriptorSetLayoutCreateFlags(),setLayoutBindings.size(),setLayoutBindings.data());

    descriptorSetLayout.resize(1);
    descriptorSetLayout[0] = device.createDescriptorSetLayout(descriptorLayout);
    //    vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout[0]);

}

void ComputePipeline::createPipelineLayout(std::vector<vk::PushConstantRange> pushConstantRanges )
{
    SAIGA_ASSERT(device);
    // Pipeline layout
    //    VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(mat4), 0);



    //        VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
    vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo(
                vk::PipelineLayoutCreateFlags(),
                descriptorSetLayout.size(),
                descriptorSetLayout.data(),
                pushConstantRanges.size(),
                pushConstantRanges.data()
                );// = vks::initializers::pipelineLayoutCreateInfo(descriptorSetLayout.data(), descriptorSetLayout.size());
    //    pPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    //    pPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    //    vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout);
    pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);

    SAIGA_ASSERT(pipelineLayout);
}

void ComputePipeline::createDescriptorPool(int maxDescriptorSets, std::vector<vk::DescriptorPoolSize> poolSizes)
{
    SAIGA_ASSERT(device);
    // descriptor pool


    vk::DescriptorPoolCreateInfo descriptorPoolInfo(vk::DescriptorPoolCreateFlags(),maxDescriptorSets,poolSizes.size(),poolSizes.data());
    descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
    SAIGA_ASSERT(descriptorPool);

    //    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, numDescriptorSets);
    //    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
}


void ComputePipeline::preparePipelines(ComputePipelineInfo &pipelineInfo, VkPipelineCache pipelineCache)
{ 
    SAIGA_ASSERT(device);

    pipelineInfo.setShader(shader);
    auto pipelineCreateInfo= pipelineInfo.createCreateInfo(pipelineLayout);
    pipeline = device.createComputePipeline(pipelineCache,pipelineCreateInfo);
    SAIGA_ASSERT(pipeline);
}



}
}
