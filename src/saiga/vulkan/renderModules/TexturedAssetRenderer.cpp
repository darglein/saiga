/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "TexturedAssetRenderer.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"
#include "saiga/assets/model/objModelLoader.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#error OpenGL was included somewhere.
#endif

namespace Saiga {
namespace Vulkan {



void TexturedAssetRenderer::destroy()
{
    Pipeline::destroy();
    uniformBufferVS.destroy();
}
void TexturedAssetRenderer::bind(vk::CommandBuffer cmd)
{
//    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,pipelineLayout,0,descriptorSet,nullptr);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,pipeline);
}

void TexturedAssetRenderer::bindTexture(vk::CommandBuffer cmd, vk::DescriptorSet ds)
{
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,pipelineLayout,0,ds,nullptr);

}

void TexturedAssetRenderer::pushModel(VkCommandBuffer cmd, mat4 model)
{
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mat4), &model[0][0]);
}

void TexturedAssetRenderer::updateUniformBuffers(glm::mat4 view, glm::mat4 proj)
{
    // Vertex shader
    uboVS.projection = proj;
    uboVS.modelview = view;
    uboVS.lightPos = vec4(5,5,5,0);



    VK_CHECK_RESULT(uniformBufferVS.map());
    memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
    uniformBufferVS.unmap();

}

void TexturedAssetRenderer::updateUniformBuffers(vk::CommandBuffer cmd, glm::mat4 view, glm::mat4 proj)
{
    uboVS.projection = proj;
    uboVS.modelview = view;
    uboVS.lightPos = vec4(5,5,5,0);
    cmd.updateBuffer(uniformBufferVS.buffer,0,sizeof(uboVS),&uboVS);
}

void TexturedAssetRenderer::init(VulkanBase &vulkanDevice, VkRenderPass renderPass)
{
    this->base = &vulkanDevice;
    this->device = vulkanDevice.device;

    uint32_t numUniformBuffers = 1;

    // Vertex shader uniform buffer block
    VK_CHECK_RESULT(vulkanDevice.createBuffer(
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &uniformBufferVS,
                        sizeof(uboVS),
                        &uboVS));


    createDescriptorSetLayout({
                                  vk::DescriptorSetLayoutBinding{ 7,vk::DescriptorType::eUniformBuffer,1,vk::ShaderStageFlagBits::eVertex },
                                  vk::DescriptorSetLayoutBinding{ 11,vk::DescriptorType::eCombinedImageSampler,1,vk::ShaderStageFlagBits::eFragment},
                              });


    createPipelineLayout({
                             vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex,0,sizeof(mat4))
                         });





    // Load all shaders.
    // Note: The shader type is deduced from the ending.
    shaderPipeline.load(
                device,{
                    "vulkan/texture.vert",
                    "vulkan/texture.frag"
                });

    // We use the default pipeline with "VertexNC" input vertices.
    PipelineInfo info;
    info.addVertexInfo<VertexType>();
    preparePipelines(info,vulkanDevice.pipelineCache,renderPass);
}

vk::DescriptorSet TexturedAssetRenderer::createAndUpdateDescriptorSet(Texture &texture)
{
//    vk::DescriptorSet descriptorSet = device.allocateDescriptorSets(
//                vk::DescriptorSetAllocateInfo(descriptorPool,descriptorSetLayout.size(),descriptorSetLayout.data())
//                )[0];

//    auto set = base->descriptorPool.allocateDescriptorSet(descriptorSetLayout[0]);
    auto set = createDescriptorSet();



    vk::DescriptorImageInfo descriptorInfoTexture = texture.getDescriptorInfo();



    vk::DescriptorBufferInfo descriptorInfo = uniformBufferVS.descriptor;
    device.updateDescriptorSets({
                                    vk::WriteDescriptorSet(set,7,0,1,vk::DescriptorType::eUniformBuffer,nullptr,&descriptorInfo,nullptr),
                                    vk::WriteDescriptorSet(set,11,0,1,vk::DescriptorType::eCombinedImageSampler,&descriptorInfoTexture,nullptr,nullptr),
                                },nullptr);
    return set;
}





}
}
