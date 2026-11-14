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
/**
 * The attachment formats a graphics pipeline renders to (dynamic rendering, replaces the render pass).
 */
struct SAIGA_VULKAN_API RenderingFormats
{
    std::vector<vk::Format> color;
    vk::Format depth   = vk::Format::eUndefined;
    vk::Format stencil = vk::Format::eUndefined;
};

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

    // Blending is off by default. If enabled: alpha blending ("over") of color and alpha, so that the alpha of an
    // attachment stays correct (an opaque source writes alpha = 1) when it is blended again later.
    vk::PipelineColorBlendAttachmentState blendAttachmentState = {
        false,
        vk::BlendFactor::eSrcAlpha,
        vk::BlendFactor::eOneMinusSrcAlpha,
        vk::BlendOp::eAdd,
        vk::BlendFactor::eOne,
        vk::BlendFactor::eOneMinusSrcAlpha,
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



    // The per-vertex attributes of the vertex type in binding 0.
    template <typename VertexType>
    void addVertexInfo();

    /**
     * Adds a vertex buffer binding with its attributes, e.g. per-instance data in binding 1:
     *
     *   info.addVertexBinding(1, sizeof(InstanceData), vk::VertexInputRate::eInstance,
     *                         {{4, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(InstanceData, offset)}});
     *
     * The attribute descriptions are {location, binding, format, offset}. Bind the buffer with
     * VertexBuffer::bind(cmd, binding) and draw with an instance count.
     */
    void addVertexBinding(uint32_t binding, uint32_t stride, vk::VertexInputRate inputRate,
                          const std::vector<vk::VertexInputAttributeDescription>& attributes);

    void addShaders(Saiga::Vulkan::GraphicsShaderPipeline& shaders);

    // The returned struct points into this object, which must stay alive until the pipeline is created.
    vk::GraphicsPipelineCreateInfo createCreateInfo(vk::PipelineLayout pipelineLayout, const RenderingFormats& formats);

   private:
    RenderingFormats renderingFormats;
    vk::PipelineRenderingCreateInfo renderingInfo;
    std::vector<vk::PipelineColorBlendAttachmentState> blendAttachmentStates;
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    vk::PipelineVertexInputStateCreateInfo vi;
    std::vector<vk::VertexInputBindingDescription> vertexInputBindings;
    std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes;
    vk::PipelineDynamicStateCreateInfo dynamicState;
};

template <typename VertexType>
void PipelineInfo::addVertexInfo()
{
    VKVertexAttribBinder<VertexType> va;
    vk::VertexInputBindingDescription binding;
    std::vector<vk::VertexInputAttributeDescription> attributes;
    va.getVKAttribs(binding, attributes);
    addVertexBinding(binding.binding, binding.stride, binding.inputRate, attributes);
}


}  // namespace Vulkan
}  // namespace Saiga
