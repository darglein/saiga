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


void Pipeline::preparePipelines(VkPipelineCache pipelineCache, VkRenderPass renderPass)
{

    // Rendering
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
            vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0,	VK_FALSE);

    VkPipelineRasterizationStateCreateInfo rasterizationState =
            vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);

    VkPipelineColorBlendAttachmentState blendAttachmentState =
            vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);

    VkPipelineColorBlendStateCreateInfo colorBlendState =
            vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

    VkPipelineDepthStencilStateCreateInfo depthStencilState =
            vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);

    VkPipelineViewportStateCreateInfo viewportState =
            vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

    VkPipelineMultisampleStateCreateInfo multisampleState =
            vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);

    std::vector<VkDynamicState> dynamicStateEnables = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState =
            vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);


    VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);

    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;

    vk::VertexInputBindingDescription vi_binding;
    std::vector<vk::VertexInputAttributeDescription> vi_attribs;

    VKVertexAttribBinder<VertexNC> va;
    va.getVKAttribs(vi_binding,vi_attribs);

    vk::PipelineVertexInputStateCreateInfo vi;
    //    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    //    vi.pNext = NULL;
    //    vi.flags = 0;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vi_binding;
    vi.vertexAttributeDescriptionCount = vi_attribs.size();
    vi.pVertexAttributeDescriptions = vi_attribs.data();

    VkPipelineVertexInputStateCreateInfo vertexInputState = vi;
    pipelineCreateInfo.pVertexInputState = &vertexInputState;

    shaderPipeline.load(device,{
                            "vulkan/scene.vert",
                            "vulkan/scene.frag"
                        });
    shaderPipeline.addToPipeline(pipelineCreateInfo);

    vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline);
}

}
}
