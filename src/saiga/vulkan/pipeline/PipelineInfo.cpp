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
                                                              vk::RenderPass renderPass)
{
    dynamicState.dynamicStateCount = dynamicStateEnables.size();
    dynamicState.pDynamicStates    = dynamicStateEnables.data();


    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &vertexInputBindings;
    vi.vertexAttributeDescriptionCount = vertexInputAttributes.size();
    vi.pVertexAttributeDescriptions    = vertexInputAttributes.data();


    colorBlendState.pAttachments    = &blendAttachmentState;
    colorBlendState.attachmentCount = 1;

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo(
        vk::PipelineCreateFlags(), shaderStages.size(), shaderStages.data(), &vi, &inputAssemblyState,
        &tessellationState, &viewportState, &rasterizationState, &multisampleState, &depthStencilState,
        &colorBlendState, &dynamicState, pipelineLayout, renderPass, 0, vk::Pipeline(), 0);
    return pipelineCreateInfo;
}



}  // namespace Vulkan
}  // namespace Saiga
