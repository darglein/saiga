/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Shader/ShaderPipeline.h"
#include "saiga/vulkan/Vertex.h"
#include "saiga/vulkan/svulkan.h"

namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API PipelineInfo
{
   public:
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState = {vk::PipelineInputAssemblyStateCreateFlags(),
                                                                   vk::PrimitiveTopology::eTriangleList, false};

    vk::PipelineRasterizationStateCreateInfo rasterizationState = {vk::PipelineRasterizationStateCreateFlags(),
                                                                   false,
                                                                   false,
                                                                   vk::PolygonMode::eFill,
                                                                   vk::CullModeFlagBits::eBack,
                                                                   vk::FrontFace::eCounterClockwise,
                                                                   false,
                                                                   0,
                                                                   0,
                                                                   0,
                                                                   1};

    vk::PipelineDepthStencilStateCreateInfo depthStencilState = {vk::PipelineDepthStencilStateCreateFlags(),
                                                                 true,
                                                                 true,
                                                                 vk::CompareOp::eLessOrEqual,
                                                                 false,
                                                                 false,
                                                                 vk::StencilOpState(),
                                                                 vk::StencilOpState(),
                                                                 0,
                                                                 0};

    vk::PipelineColorBlendAttachmentState blendAttachmentState = {
        false,
        vk::BlendFactor::eSrcAlpha,
        vk::BlendFactor::eOneMinusSrcAlpha,
        vk::BlendOp::eAdd,
        vk::BlendFactor::eOneMinusSrcAlpha,
        vk::BlendFactor::eZero,
        vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA};

    vk::PipelineColorBlendStateCreateInfo colorBlendState = {
        vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eClear, 1, &blendAttachmentState, {{0, 0, 0, 0}}};



    vk::PipelineViewportStateCreateInfo viewportState = {vk::PipelineViewportStateCreateFlags(), 1, nullptr, 1,
                                                         nullptr};

    vk::PipelineMultisampleStateCreateInfo multisampleState = {
        vk::PipelineMultisampleStateCreateFlags(), vk::SampleCountFlagBits::e1, 0, 0, nullptr, 0, 0};

    std::vector<vk::DynamicState> dynamicStateEnables = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    vk::PipelineTessellationStateCreateInfo tessellationState = {vk::PipelineTessellationStateCreateFlags(), 0};



    template <typename VertexType>
    void addVertexInfo();

    void addShaders(Saiga::Vulkan::GraphicsShaderPipeline& shaders);

    vk::GraphicsPipelineCreateInfo createCreateInfo(vk::PipelineLayout pipelineLayout, vk::RenderPass renderPass);

   private:
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    vk::PipelineVertexInputStateCreateInfo vi;
    vk::VertexInputBindingDescription vertexInputBindings;
    std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes;
    vk::PipelineDynamicStateCreateInfo dynamicState;
};

template <typename VertexType>
void PipelineInfo::addVertexInfo()
{
    VKVertexAttribBinder<VertexType> va;
    va.getVKAttribs(vertexInputBindings, vertexInputAttributes);
}


}  // namespace Vulkan
}  // namespace Saiga
