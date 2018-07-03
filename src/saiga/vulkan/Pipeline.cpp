/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Pipeline.h"
#include "saiga/vulkan/VulkanInitializers.hpp"


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

    descriptorSetLayout[0] = device.createDescriptorSetLayout(descriptorLayout);
//    vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout[0]);

}

void Pipeline::createPipelineLayout()
{
    // Pipeline layout
//    VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(mat4), 0);

    std::vector<vk::PushConstantRange> pushConstantRanges = {
    vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex,0,sizeof(mat4))
    };

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

}
}
