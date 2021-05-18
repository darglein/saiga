/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ProceduralSkybox.h"

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
void ProceduralSkybox::destroy()
{
    Pipeline::destroy();
}

void ProceduralSkybox::renderTexture(vk::CommandBuffer cmd, StaticDescriptorSet texture, vec2 position, vec2 size)
{
    bindDescriptorSet(cmd, texture);
    vk::Viewport vp(position[0], position[1], size[0], size[1]);
    cmd.setViewport(0, vp);
    blitMesh.render(cmd);
}



void ProceduralSkybox::init(VulkanBase& vulkanDevice, VkRenderPass renderPass)
{
    PipelineBase::init(vulkanDevice, 1);
    addDescriptorSetLayout({{
        0,
        {11, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
    }});
    addPushConstantRange({vk::ShaderStageFlagBits::eVertex, 0, sizeof(mat4)});
    shaderPipeline.load(device, {"vulkan/skybox.vert", "vulkan/skybox.frag"});
    PipelineInfo info;
    info.addVertexInfo<VertexType>();
    info.rasterizationState.cullMode      = vk::CullModeFlagBits::eNone;
    info.blendAttachmentState.blendEnable = true;
    create(renderPass, info);

    SAIGA_EXIT_ERROR("todo");
    //    blitMesh.createFullscreenQuad();
    blitMesh.init(vulkanDevice);
}

StaticDescriptorSet ProceduralSkybox::createAndUpdateDescriptorSet(Texture& texture)
{
    //    vk::DescriptorSet descriptorSet = device.allocateDescriptorSets(
    //                vk::DescriptorSetAllocateInfo(descriptorPool,descriptorSetLayout.size(),descriptorSetLayout.data())
    //                )[0];

    //    auto set = base->descriptorPool.allocateDescriptorSet(descriptorSetLayout[0]);
    auto set = createDescriptorSet();

    set.assign(0, &texture);

    return set;
}



}  // namespace Vulkan
}  // namespace Saiga
