/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "PointCloudRenderer.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"
#include "saiga/assets/model/objModelLoader.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#error OpenGL was included somewhere.
#endif

namespace Saiga {
namespace Vulkan {



void PointCloudRenderer::destroy()
{
    Pipeline::destroy();
    uniformBufferVS.destroy();
}
void PointCloudRenderer::bind(vk::CommandBuffer cmd)
{
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,pipelineLayout,0,descriptorSet,nullptr);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,pipeline);
}

void PointCloudRenderer::pushModel(VkCommandBuffer cmd, mat4 model)
{
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mat4), &model[0][0]);
}



void PointCloudRenderer::updateUniformBuffers(vk::CommandBuffer cmd, glm::mat4 view, glm::mat4 proj)
{
    uboVS.projection = proj;
    uboVS.modelview = view;
    uboVS.lightPos = vec4(5,5,5,0);
      cmd.updateBuffer(uniformBufferVS.buffer,0,sizeof(uboVS),&uboVS);
}

void PointCloudRenderer::init(VulkanBase &vulkanDevice, VkRenderPass renderPass, float pointSize)
{
    this->base = &vulkanDevice;
    this->device = vulkanDevice.device;


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
    shaderPipeline.loadGLSL(
                device,{
                    {"vulkan/point.vert",   vk::ShaderStageFlagBits::eVertex,   "#define POINT_SIZE " + std::to_string(pointSize)},
                    {"vulkan/point.frag",    vk::ShaderStageFlagBits::eFragment, ""}
                });

    // We use the default pipeline with "VertexNC" input vertices.
    PipelineInfo info;
    info.inputAssemblyState.topology = vk::PrimitiveTopology::ePointList;
    info.addVertexInfo<VertexType>();
    preparePipelines(info,vulkanDevice.pipelineCache,renderPass);
}





}
}
