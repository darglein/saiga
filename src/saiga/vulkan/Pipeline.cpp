/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Pipeline.h"
#include "saiga/vulkan/VulkanInitializers.hpp"
#include "saiga/vulkan/Vertex.h"

namespace Saiga {
namespace Vulkan {


void Pipeline::destroy()
{
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    for(auto& l :descriptorSetLayout)
        vkDestroyDescriptorSetLayout(device, l, nullptr);
    vkDestroyDescriptorPool(device,descriptorPool,nullptr);
    shaderPipeline.destroy(device);

}




void Pipeline::createDescriptorSetLayout(std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings)
{
    //    VkDescriptorSetLayoutCreateInfo descriptorLayout =
    //            vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    vk::DescriptorSetLayoutCreateInfo descriptorLayout(vk::DescriptorSetLayoutCreateFlags(),setLayoutBindings.size(),setLayoutBindings.data());

    descriptorSetLayout.resize(1);
    descriptorSetLayout[0] = device.createDescriptorSetLayout(descriptorLayout);
    //    vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout[0]);

}

void Pipeline::createPipelineLayout(std::vector<vk::PushConstantRange> pushConstantRanges )
{
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

void Pipeline::createDescriptorPool(int maxDescriptorSets, std::vector<vk::DescriptorPoolSize> poolSizes)
{
    // descriptor pool


    vk::DescriptorPoolCreateInfo descriptorPoolInfo(vk::DescriptorPoolCreateFlags(),maxDescriptorSets,poolSizes.size(),poolSizes.data());
    descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
    SAIGA_ASSERT(descriptorPool);

    //    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, numDescriptorSets);
    //    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
}


void Pipeline::preparePipelines(PipelineInfo& pipelineInfo, VkPipelineCache pipelineCache, vk::RenderPass renderPass)
{ 
    pipelineInfo.addShaders(shaderPipeline);
    auto pipelineCreateInfo= pipelineInfo.createCreateInfo(pipelineLayout,renderPass);
    pipeline = device.createGraphicsPipeline(pipelineCache,pipelineCreateInfo);
    SAIGA_ASSERT(pipeline);
}

}
}
