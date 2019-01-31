/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "TextureDisplay.h"

#include "saiga/core/model/objModelLoader.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

namespace Saiga
{
namespace Vulkan
{
void TextureDisplay::destroy()
{
    Pipeline::destroy();
}

void TextureDisplay::renderTexture(vk::CommandBuffer cmd, vk::DescriptorSet texture, vec2 position, vec2 size)
{
    bindDescriptorSets(cmd, texture);
    vk::Viewport vp(position[0], position[1], size[0], size[1]);
    cmd.setViewport(0, vp);
    blitMesh.render(cmd);
}



void TextureDisplay::init(VulkanBase& vulkanDevice, VkRenderPass renderPass)
{
    PipelineBase::init(vulkanDevice, 1);
    addDescriptorSetLayout({
        {11, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
    });
    addPushConstantRange({vk::ShaderStageFlagBits::eVertex, 0, sizeof(mat4)});
    shaderPipeline.load(device, {"vulkan/blit.vert", "vulkan/blit.frag"});
    PipelineInfo info;
    info.addVertexInfo<VertexType>();
    info.rasterizationState.cullMode      = vk::CullModeFlagBits::eNone;
    info.blendAttachmentState.blendEnable = true;
    create(renderPass, info);

    blitMesh.createFullscreenQuad();
    blitMesh.init(vulkanDevice);
}

vk::DescriptorSet TextureDisplay::createAndUpdateDescriptorSet(Texture& texture)
{
    //    vk::DescriptorSet descriptorSet = device.allocateDescriptorSets(
    //                vk::DescriptorSetAllocateInfo(descriptorPool,descriptorSetLayout.size(),descriptorSetLayout.data())
    //                )[0];

    //    auto set = base->descriptorPool.allocateDescriptorSet(descriptorSetLayout[0]);
    auto set = createDescriptorSet();



    vk::DescriptorImageInfo descriptorInfoTexture = texture.getDescriptorInfo();



    device.updateDescriptorSets(
        {
            vk::WriteDescriptorSet(set, 11, 0, 1, vk::DescriptorType::eCombinedImageSampler, &descriptorInfoTexture,
                                   nullptr, nullptr),
        },
        nullptr);
    return set;
}



}  // namespace Vulkan
}  // namespace Saiga
