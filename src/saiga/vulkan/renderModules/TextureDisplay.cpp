/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "TextureDisplay.h"

#include "saiga/core/model/model_from_shape.h"
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
void TextureDisplay::destroy()
{
    Pipeline::destroy();
    shaderPipeline.destroy();
}

void TextureDisplay::renderTexture(vk::CommandBuffer cmd, DescriptorSet& descriptor, vec2 position, vec2 size,
                                   std::optional<vk::Viewport> restoreViewport)
{
    bindDescriptorSet(cmd, descriptor);
    cmd.setViewport(0, vk::Viewport(position[0], position[1], size[0], size[1], 0.0f, 1.0f));
    blitMesh.render(cmd);
    if (restoreViewport) cmd.setViewport(0, *restoreViewport);
}



void TextureDisplay::init(VulkanBase& vulkanDevice, const RenderingFormats& formats)
{
    PipelineBase::init(vulkanDevice, 1);
    addDescriptorSetLayout({
        {0, {11, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}},
    });
    addPushConstantRange({vk::ShaderStageFlagBits::eVertex, 0, sizeof(mat4)});
    shaderPipeline.load(device, {"vulkan/blit.vert", "vulkan/blit.frag"});
    PipelineInfo info;
    info.addVertexInfo<VertexType>();
    info.rasterizationState.cullMode      = vk::CullModeFlagBits::eNone;
    info.blendAttachmentState.blendEnable = VK_TRUE;
    create(formats, info);

    static_cast<TriangleMesh<VertexType, uint32_t>&>(blitMesh) = FullScreenQuad().Mesh<VertexType, uint32_t>();
    blitMesh.init(vulkanDevice);
}

StaticDescriptorSet TextureDisplay::createAndUpdateDescriptorSet(Texture& texture)
{
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
