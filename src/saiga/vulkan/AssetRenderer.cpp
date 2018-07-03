/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "AssetRenderer.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"
#include "saiga/assets/model/objModelLoader.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#error OpenGL was included somewhere.
#endif

namespace Saiga {
namespace Vulkan {



void AssetRenderer::destroy()
{
    Pipeline::destroy();
    uniformBufferVS.destroy();
}

void AssetRenderer::bind(vk::CommandBuffer cmd)
{
//    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSet.size(), descriptorSet.data(), 0, nullptr);

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,pipelineLayout,0,descriptorSet,nullptr);
    //    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
}

void AssetRenderer::pushModel(VkCommandBuffer cmd, mat4 model)
{
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mat4), &model[0][0]);
}

void AssetRenderer::updateUniformBuffers(glm::mat4 view, glm::mat4 proj)
{
    // Vertex shader
    uboVS.projection = proj;
    uboVS.modelview = view;
    uboVS.lightPos = vec4(5,5,5,0);



    VK_CHECK_RESULT(uniformBufferVS.map());
    memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
    uniformBufferVS.unmap();
}

void AssetRenderer::init(vks::VulkanDevice *vulkanDevice, VkPipelineCache pipelineCache, VkRenderPass renderPass)
{

    this->device = vulkanDevice->logicalDevice;

    prepareUniformBuffers(vulkanDevice);
    setupLayoutsAndDescriptors();
    preparePipelines(device,pipelineCache,renderPass);
}



void AssetRenderer::prepareUniformBuffers(vks::VulkanDevice *vulkanDevice)
{
    // Vertex shader uniform buffer block
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &uniformBufferVS,
                        sizeof(uboVS),
                        &uboVS));


}


void AssetRenderer::setupLayoutsAndDescriptors()
{


    int numDescriptorSets = 1;
    descriptorSetLayout.resize(numDescriptorSets);
    descriptorSet.resize(numDescriptorSets);

    {
        createDescriptorSetLayout({
                                      { 5,vk::DescriptorType::eUniformBuffer,1,vk::ShaderStageFlagBits::eVertex },
                                      { 7,vk::DescriptorType::eUniformBuffer,1,vk::ShaderStageFlagBits::eVertex }
                                  });
    }

    {
       createPipelineLayout();
}


    {
        // descriptor pool
        std::vector<VkDescriptorPoolSize> poolSizes = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
            //        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1)
        };
        VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, numDescriptorSets);
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
    }


    {
        // sets the descriptor to the uniform buffer
        vk::DescriptorSetAllocateInfo allocInfo(descriptorPool,descriptorSetLayout.size(),descriptorSetLayout.data()) ;//=	vks::initializers::descriptorSetAllocateInfo(descriptorPool, descriptorSetLayout.data(), descriptorSetLayout.size());
//        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, descriptorSet.data()));
        descriptorSet = device.allocateDescriptorSets(allocInfo);

        std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
            vks::initializers::writeDescriptorSet(descriptorSet[0], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5, &uniformBufferVS.descriptor),
            vks::initializers::writeDescriptorSet(descriptorSet[0], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 7, &uniformBufferVS.descriptor),
//            vks::initializers::writeDescriptorSet(descriptorSet[1], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, &uniformBufferVS.descriptor)
        };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
    }
}

void AssetRenderer::preparePipelines(VkDevice device, VkPipelineCache pipelineCache, VkRenderPass renderPass)
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

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline));
}



}
}
