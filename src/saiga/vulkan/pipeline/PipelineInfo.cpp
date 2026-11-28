/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "PipelineInfo.h"

#include <algorithm>

namespace Saiga
{
namespace Vulkan
{
namespace
{
vk::CompareOp toVk(CompareOp op)
{
    switch (op)
    {
        case CompareOp::Never:
            return vk::CompareOp::eNever;
        case CompareOp::Less:
            return vk::CompareOp::eLess;
        case CompareOp::Equal:
            return vk::CompareOp::eEqual;
        case CompareOp::LessOrEqual:
            return vk::CompareOp::eLessOrEqual;
        case CompareOp::Greater:
            return vk::CompareOp::eGreater;
        case CompareOp::NotEqual:
            return vk::CompareOp::eNotEqual;
        case CompareOp::GreaterOrEqual:
            return vk::CompareOp::eGreaterOrEqual;
        case CompareOp::Always:
            return vk::CompareOp::eAlways;
    }
    return vk::CompareOp::eAlways;
}

vk::StencilOp toVk(StencilOp op)
{
    switch (op)
    {
        case StencilOp::Keep:
            return vk::StencilOp::eKeep;
        case StencilOp::Zero:
            return vk::StencilOp::eZero;
        case StencilOp::Replace:
            return vk::StencilOp::eReplace;
        case StencilOp::IncrementAndClamp:
            return vk::StencilOp::eIncrementAndClamp;
        case StencilOp::DecrementAndClamp:
            return vk::StencilOp::eDecrementAndClamp;
        case StencilOp::Invert:
            return vk::StencilOp::eInvert;
        case StencilOp::IncrementAndWrap:
            return vk::StencilOp::eIncrementAndWrap;
        case StencilOp::DecrementAndWrap:
            return vk::StencilOp::eDecrementAndWrap;
    }
    return vk::StencilOp::eKeep;
}

vk::BlendFactor toVk(BlendFactor f)
{
    switch (f)
    {
        case BlendFactor::Zero:
            return vk::BlendFactor::eZero;
        case BlendFactor::One:
            return vk::BlendFactor::eOne;
        case BlendFactor::SrcColor:
            return vk::BlendFactor::eSrcColor;
        case BlendFactor::OneMinusSrcColor:
            return vk::BlendFactor::eOneMinusSrcColor;
        case BlendFactor::DstColor:
            return vk::BlendFactor::eDstColor;
        case BlendFactor::OneMinusDstColor:
            return vk::BlendFactor::eOneMinusDstColor;
        case BlendFactor::SrcAlpha:
            return vk::BlendFactor::eSrcAlpha;
        case BlendFactor::OneMinusSrcAlpha:
            return vk::BlendFactor::eOneMinusSrcAlpha;
        case BlendFactor::DstAlpha:
            return vk::BlendFactor::eDstAlpha;
        case BlendFactor::OneMinusDstAlpha:
            return vk::BlendFactor::eOneMinusDstAlpha;
    }
    return vk::BlendFactor::eOne;
}

vk::BlendOp toVk(BlendOp op)
{
    switch (op)
    {
        case BlendOp::Add:
            return vk::BlendOp::eAdd;
        case BlendOp::Subtract:
            return vk::BlendOp::eSubtract;
        case BlendOp::ReverseSubtract:
            return vk::BlendOp::eReverseSubtract;
        case BlendOp::Min:
            return vk::BlendOp::eMin;
        case BlendOp::Max:
            return vk::BlendOp::eMax;
    }
    return vk::BlendOp::eAdd;
}

vk::StencilOpState toVk(const StencilFaceState& s)
{
    return vk::StencilOpState(toVk(s.failOp), toVk(s.passOp), toVk(s.depthFailOp), toVk(s.compareOp), s.compareMask,
                              s.writeMask, 0);
}
}  // namespace

void PipelineInfo::setState(const PipelineState& state)
{
    inputAssemblyState.primitiveRestartEnable = state.primitiveRestart;

    const RasterizationState& r = state.rasterization;
    switch (r.polygonMode)
    {
        case PolygonMode::Fill:
            rasterizationState.polygonMode = vk::PolygonMode::eFill;
            break;
        case PolygonMode::Line:
            rasterizationState.polygonMode = vk::PolygonMode::eLine;
            break;
        case PolygonMode::Point:
            rasterizationState.polygonMode = vk::PolygonMode::ePoint;
            break;
    }
    switch (r.cullMode)
    {
        case CullMode::None:
            rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
            break;
        case CullMode::Front:
            rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
            break;
        case CullMode::Back:
            rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
            break;
        case CullMode::FrontAndBack:
            rasterizationState.cullMode = vk::CullModeFlagBits::eFrontAndBack;
            break;
    }
    rasterizationState.frontFace =
        r.frontFace == FrontFace::CounterClockwise ? vk::FrontFace::eCounterClockwise : vk::FrontFace::eClockwise;
    rasterizationState.depthBiasEnable         = r.depthBias;
    rasterizationState.depthBiasConstantFactor = r.depthBiasConstant;
    rasterizationState.depthBiasSlopeFactor    = r.depthBiasSlope;

    const DepthStencilState& d          = state.depthStencil;
    depthStencilState.depthTestEnable   = d.depthTest;
    depthStencilState.depthWriteEnable  = d.depthWrite;
    depthStencilState.depthCompareOp    = toVk(d.depthCompare);
    depthStencilState.stencilTestEnable = d.stencilTest;
    depthStencilState.front             = toVk(d.front);
    depthStencilState.back              = toVk(d.back);
    if (d.stencilTest && std::find(dynamicStateEnables.begin(), dynamicStateEnables.end(),
                                   vk::DynamicState::eStencilReference) == dynamicStateEnables.end())
    {
        dynamicStateEnables.push_back(vk::DynamicState::eStencilReference);
    }

    const BlendState& b                      = state.blend;
    blendAttachmentState.blendEnable         = b.enable;
    blendAttachmentState.srcColorBlendFactor = toVk(b.srcColor);
    blendAttachmentState.dstColorBlendFactor = toVk(b.dstColor);
    blendAttachmentState.colorBlendOp        = toVk(b.colorOp);
    blendAttachmentState.srcAlphaBlendFactor = toVk(b.srcAlpha);
    blendAttachmentState.dstAlphaBlendFactor = toVk(b.dstAlpha);
    blendAttachmentState.alphaBlendOp        = toVk(b.alphaOp);
    vk::ColorComponentFlags mask;
    if (b.colorWriteMask & ColorComponentR) mask |= vk::ColorComponentFlagBits::eR;
    if (b.colorWriteMask & ColorComponentG) mask |= vk::ColorComponentFlagBits::eG;
    if (b.colorWriteMask & ColorComponentB) mask |= vk::ColorComponentFlagBits::eB;
    if (b.colorWriteMask & ColorComponentA) mask |= vk::ColorComponentFlagBits::eA;
    blendAttachmentState.colorWriteMask = mask;
}

void PipelineInfo::addShaders(GraphicsShaderPipeline& shaders)
{
    //    shaderStages.clear();
    //    for (auto& s : shaders.modules)
    //    {
    //        shaderStages.push_back(s.createPipelineInfo());
    //    }
}

void PipelineInfo::addVertexBinding(uint32_t binding, uint32_t stride, vk::VertexInputRate inputRate,
                                    const std::vector<vk::VertexInputAttributeDescription>& attributes)
{
    for (auto& b : vertexInputBindings) SAIGA_ASSERT(b.binding != binding, "The vertex binding is already used");
    vertexInputBindings.emplace_back(binding, stride, inputRate);
    for (auto a : attributes)
    {
        SAIGA_ASSERT(a.binding == binding, "The attribute belongs to another binding");
        vertexInputAttributes.push_back(a);
    }
}

vk::GraphicsPipelineCreateInfo PipelineInfo::createCreateInfo(vk::PipelineLayout pipelineLayout,
                                                              const RenderingFormats& formats)
{
    dynamicState.dynamicStateCount = dynamicStateEnables.size();
    dynamicState.pDynamicStates    = dynamicStateEnables.data();

    // Pipelines without vertex input (e.g. full screen passes that generate the vertices in the shader)
    vi.vertexBindingDescriptionCount   = vertexInputBindings.size();
    vi.pVertexBindingDescriptions      = vertexInputBindings.data();
    vi.vertexAttributeDescriptionCount = vertexInputAttributes.size();
    vi.pVertexAttributeDescriptions    = vertexInputAttributes.data();

    // One blend state per color attachment (none for depth only passes).
    blendAttachmentStates.assign(formats.color.size(), blendAttachmentState);
    colorBlendState.pAttachments    = blendAttachmentStates.data();
    colorBlendState.attachmentCount = blendAttachmentStates.size();

    // Multisampled attachments (e.g. the MSAA main pass of VulkanForwardRenderer)
    if (formats.samples != vk::SampleCountFlagBits::e1) multisampleState.rasterizationSamples = formats.samples;

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
