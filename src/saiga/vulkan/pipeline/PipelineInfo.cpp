/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "PipelineInfo.h"

namespace Saiga
{
namespace Vulkan
{
void PipelineInfo::addShaders(GraphicsShaderPipeline& shaders)
{
    //    shaderStages.clear();
    //    for (auto& s : shaders.modules)
    //    {
    //        shaderStages.push_back(s.createPipelineInfo());
    //    }
}

vk::GraphicsPipelineCreateInfo PipelineInfo::createCreateInfo(vk::PipelineLayout pipelineLayout,
                                                              const RenderingFormats& formats)
{
    dynamicState.dynamicStateCount = dynamicStateEnables.size();
    dynamicState.pDynamicStates    = dynamicStateEnables.data();

    // Pipelines without vertex input (e.g. full screen passes that generate the vertices in the shader)
    vi.vertexBindingDescriptionCount   = hasVertexInfo ? 1 : 0;
    vi.pVertexBindingDescriptions      = hasVertexInfo ? &vertexInputBindings : nullptr;
    vi.vertexAttributeDescriptionCount = vertexInputAttributes.size();
    vi.pVertexAttributeDescriptions    = vertexInputAttributes.data();

    // One blend state per color attachment (none for depth only passes).
    blendAttachmentStates.assign(formats.color.size(), blendAttachmentState);
    colorBlendState.pAttachments    = blendAttachmentStates.data();
    colorBlendState.attachmentCount = blendAttachmentStates.size();

    // Dynamic rendering: the attachment formats instead of a render pass.
    renderingFormats                      = formats;
    renderingInfo                         = vk::PipelineRenderingCreateInfo();
    renderingInfo.colorAttachmentCount    = renderingFormats.color.size();
    renderingInfo.pColorAttachmentFormats = renderingFormats.color.data();
    renderingInfo.depthAttachmentFormat   = renderingFormats.depth;
    renderingInfo.stencilAttachmentFormat = renderingFormats.stencil;

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo(
        vk::PipelineCreateFlags(), shaderStages.size(), shaderStages.data(), &vi, &inputAssemblyState,
        &tessellationState, &viewportState, &rasterizationState, &multisampleState, &depthStencilState,
        &colorBlendState, &dynamicState, pipelineLayout, nullptr, 0, vk::Pipeline(), 0);
    pipelineCreateInfo.pNext = &renderingInfo;
    return pipelineCreateInfo;
}



}  // namespace Vulkan
}  // namespace Saiga
