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
    uniformBufferVS2.destroy();
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

    VK_CHECK_RESULT(uniformBufferVS2.map());
    memcpy(uniformBufferVS2.mapped, &uboVS.projection, sizeof(mat4));
    uniformBufferVS2.unmap();
}

void AssetRenderer::init(vks::VulkanDevice *vulkanDevice, VkPipelineCache pipelineCache, VkRenderPass renderPass)
{

    this->device = vulkanDevice->logicalDevice;

    prepareUniformBuffers(vulkanDevice);
    setupLayoutsAndDescriptors();

    shaderPipeline.load(device,{
                            "vulkan/scene.vert",
                            "vulkan/scene.frag"
                        });

    preparePipelines(pipelineCache,renderPass);
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

    VK_CHECK_RESULT(vulkanDevice->createBuffer(
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &uniformBufferVS2,
                        sizeof(mat4),
                        nullptr));


}


void AssetRenderer::setupLayoutsAndDescriptors()
{
    createDescriptorSetLayout({
                                  { 5,vk::DescriptorType::eUniformBuffer,1,vk::ShaderStageFlagBits::eVertex },
                                  { 7,vk::DescriptorType::eUniformBuffer,1,vk::ShaderStageFlagBits::eVertex },
                              });


    createPipelineLayout({
                             vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex,0,sizeof(mat4))
                         });


    createDescriptorPool(
                1,{
                    {vk::DescriptorType::eUniformBuffer, 2}
                });



    descriptorSet = device.allocateDescriptorSets(
                vk::DescriptorSetAllocateInfo(descriptorPool,descriptorSetLayout.size(),descriptorSetLayout.data())
                );


    vk::DescriptorBufferInfo info = uniformBufferVS.descriptor;
    vk::DescriptorBufferInfo info2 = uniformBufferVS2.descriptor;

    device.updateDescriptorSets({
                                    vk::WriteDescriptorSet(descriptorSet[0],7,0,1,vk::DescriptorType::eUniformBuffer,nullptr,&info,nullptr),
                                    vk::WriteDescriptorSet(descriptorSet[0],5,0,1,vk::DescriptorType::eUniformBuffer,nullptr,&info2,nullptr),
                                },nullptr);
}




}
}
