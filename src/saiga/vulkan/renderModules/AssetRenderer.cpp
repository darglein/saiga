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
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,pipelineLayout,0,descriptorSet,nullptr);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,pipeline);
}

void AssetRenderer::pushModel(VkCommandBuffer cmd, mat4 model)
{
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mat4), &model[0][0]);
}


void AssetRenderer::updateUniformBuffers(vk::CommandBuffer cmd, glm::mat4 view, glm::mat4 proj)
{
    uboVS.projection = proj;
    uboVS.modelview = view;
    uboVS.lightPos = vec4(5,5,5,0);
    cmd.updateBuffer(uniformBufferVS.buffer,0,sizeof(uboVS),&uboVS);
}

void AssetRenderer::init(VulkanBase &vulkanDevice, VkRenderPass renderPass)
{
    this->base = &vulkanDevice;
    this->device = vulkanDevice.device;



    // Vertex shader uniform buffer block
//    VK_CHECK_RESULT(vulkanDevice.createBuffer(
//                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
//                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
//                        &uniformBufferVS,
//                        sizeof(uboVS),
//                        &uboVS));

    uniformBufferVS.init(vulkanDevice,&uboVS,sizeof(UBOVS));


    createDescriptorSetLayout({
                                  vk::DescriptorSetLayoutBinding{ 7,vk::DescriptorType::eUniformBuffer,1,vk::ShaderStageFlagBits::eVertex },
                              });


    createPipelineLayout({
                             vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex,0,sizeof(mat4))
                         });





    descriptorSet = createDescriptorSet();

    vk::DescriptorBufferInfo descriptorInfo = uniformBufferVS.getDescriptorInfo();
    device.updateDescriptorSets({
                                    vk::WriteDescriptorSet(descriptorSet,7,0,1,vk::DescriptorType::eUniformBuffer,nullptr,&descriptorInfo,nullptr),
                                },nullptr);

    // Load all shaders.
    // Note: The shader type is deduced from the ending.
    shaderPipeline.load(
                device,{
                    "vulkan/coloredAsset.vert",
                    "vulkan/coloredAsset.frag"
                });

    // We use the default pipeline with "VertexNC" input vertices.
    PipelineInfo info;
    info.addVertexInfo<VertexNC>();
    preparePipelines(info,vulkanDevice.pipelineCache,renderPass);
}





}
}
