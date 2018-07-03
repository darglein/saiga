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


void Pipeline::preparePipelines(VkPipelineCache pipelineCache, vk::RenderPass renderPass)
{
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList, false);

    vk::PipelineRasterizationStateCreateInfo rasterizationState(
                vk::PipelineRasterizationStateCreateFlags(), false , false,
                vk::PolygonMode::eLine, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise,
                false, 0,0,0,1);

    vk::PipelineColorBlendAttachmentState blendAttachmentState(
                false, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
                vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |vk::ColorComponentFlagBits::eA);

    vk::PipelineColorBlendStateCreateInfo colorBlendState(
                vk::PipelineColorBlendStateCreateFlags(),
                false, vk::LogicOp::eClear, 1, &blendAttachmentState, { { 0, 0, 0, 0 } });


    vk::PipelineDepthStencilStateCreateInfo depthStencilState(vk::PipelineDepthStencilStateCreateFlags(),
                                                              true, true, vk::CompareOp::eLessOrEqual, false, false, vk::StencilOpState(), vk::StencilOpState(), 0, 0);


    vk::PipelineViewportStateCreateInfo viewportState(vk::PipelineViewportStateCreateFlags(),
                                                      1, nullptr, 1, nullptr);

    vk::PipelineMultisampleStateCreateInfo multisampleState(vk::PipelineMultisampleStateCreateFlags(), vk::SampleCountFlagBits::e1, 0, 0, nullptr, 0, 0);

    std::vector<vk::DynamicState> dynamicStateEnables = {

        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState(vk::PipelineDynamicStateCreateFlags(), dynamicStateEnables.size(), dynamicStateEnables.data());


    vk::PipelineTessellationStateCreateInfo tessellationState(vk::PipelineTessellationStateCreateFlags(), 0);

    vk::VertexInputBindingDescription vi_binding;
    std::vector<vk::VertexInputAttributeDescription> vi_attribs;

    VKVertexAttribBinder<VertexNC> va;
    va.getVKAttribs(vi_binding,vi_attribs);

    vk::PipelineVertexInputStateCreateInfo vi;

    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vi_binding;
    vi.vertexAttributeDescriptionCount = vi_attribs.size();
    vi.pVertexAttributeDescriptions = vi_attribs.data();


    vk::GraphicsPipelineCreateInfo pipelineCreateInfo(
                vk::PipelineCreateFlags(),
                0,
                nullptr,
                &vi,
                &inputAssemblyState,
                &tessellationState,
                &viewportState,
                &rasterizationState,
                &multisampleState,
                &depthStencilState,
                &colorBlendState,
                &dynamicState,
                pipelineLayout,
                renderPass,
                0,
                vk::Pipeline(),
                0
                );



    shaderPipeline.addToPipeline(pipelineCreateInfo);

    //    vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline);
    pipeline = device.createGraphicsPipeline(pipelineCache,pipelineCreateInfo);
    SAIGA_ASSERT(pipeline);
}

}
}
