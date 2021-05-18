/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "TexturedAssetRenderer.h"

#include "saiga/core/model/model_loader_obj.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

namespace Saiga
{
namespace Vulkan
{
void TexturedAssetRenderer::destroy()
{
    Pipeline::destroy();
    uniformBufferVS.destroy();
}

void TexturedAssetRenderer::bindTexture(vk::CommandBuffer cmd, vk::DescriptorSet ds)
{
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, ds, nullptr);
}

void TexturedAssetRenderer::pushModel(vk::CommandBuffer cmd, mat4 model)
{
    pushConstant(cmd, vk::ShaderStageFlagBits::eVertex, sizeof(mat4), model.data());
}



void TexturedAssetRenderer::updateUniformBuffers(vk::CommandBuffer cmd, mat4 view, mat4 proj)
{
    uboVS.projection = proj;
    uboVS.modelview  = view;
    uboVS.lightPos   = vec4(5, 5, 5, 0);
    uniformBufferVS.update(cmd, sizeof(UBOVS), &uboVS);
}

void TexturedAssetRenderer::init(VulkanBase& vulkanDevice, VkRenderPass renderPass, const std::string& vertShader,
                                 const std::string& fragShader)
{
    PipelineBase::init(vulkanDevice, 1);
    uniformBufferVS.init(vulkanDevice, &uboVS, sizeof(UBOVS));
    addDescriptorSetLayout({
        {0, {7, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex}},
        {1, {11, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}},
    });
    addPushConstantRange({vk::ShaderStageFlagBits::eVertex, 0, sizeof(mat4)});
    shaderPipeline.load(device, {vertShader, fragShader});
    PipelineInfo info;
    info.addVertexInfo<VertexType>();
    //    auto info2 = info;
    create(renderPass, info);
}

StaticDescriptorSet TexturedAssetRenderer::createAndUpdateDescriptorSet(Texture& texture)
{
    //    vk::DescriptorSet descriptorSet = device.allocateDescriptorSets(
    //                vk::DescriptorSetAllocateInfo(descriptorPool,descriptorSetLayout.size(),descriptorSetLayout.data())
    //                )[0];

    //    auto set = base->descriptorPool.allocateDescriptorSet(descriptorSetLayout[0]);
    auto set = createDescriptorSet();



    vk::DescriptorImageInfo descriptorInfoTexture = texture.getDescriptorInfo();



    vk::DescriptorBufferInfo descriptorInfo = uniformBufferVS.getDescriptorInfo();
    device.updateDescriptorSets(
        {
            vk::WriteDescriptorSet(set, 7, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &descriptorInfo),
            vk::WriteDescriptorSet(set, 11, 0, 1, vk::DescriptorType::eCombinedImageSampler, &descriptorInfoTexture,
                                   nullptr),
        },
        nullptr);


    return set;
}



}  // namespace Vulkan
}  // namespace Saiga
