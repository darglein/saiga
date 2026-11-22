/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Pipeline.h"

#include "saiga/vulkan/Vertex.h"
#include "saiga/vulkan/VulkanInitializers.hpp"

#include "PipelineKey.h"

namespace Saiga
{
namespace Vulkan
{
Pipeline::Pipeline() : PipelineBase(vk::PipelineBindPoint::eGraphics) {}


void Pipeline::create(const RenderingFormats& _formats, PipelineInfo _pipelineInfo)
{
    SAIGA_ASSERT(isInitialized());

    formats      = _formats;
    pipelineInfo = _pipelineInfo;

    createPipelineLayout();
    SAIGA_ASSERT(shaderPipeline.valid(), "Shader compilation failed");
    setPipeline(base->reusePipelines ? reuseKey() : std::string(),
                [this](vk::PipelineLayout layout) { return createPipeline(layout); });
}

std::string Pipeline::reuseKey()
{
    auto ci = pipelineInfo.createCreateInfo(pipelineLayout, formats);
    shaderPipeline.addToPipeline(ci);
    PipelineKeyWriter k;
    k.bytes = "G";
    k.f(ci.flags);
    k.raw(ci.stageCount);
    for (uint32_t i = 0; i < ci.stageCount; ++i) k.stage(ci.pStages[i], shaderPipeline);
    // The dynamic rendering formats (the only pNext of createCreateInfo)
    const auto* r = static_cast<const vk::PipelineRenderingCreateInfo*>(ci.pNext);
    if (!r || r->sType != vk::StructureType::ePipelineRenderingCreateInfo) return {};
    k.none(r->pNext);
    k.raw(r->viewMask);
    k.raw(r->colorAttachmentCount);
    for (uint32_t i = 0; i < r->colorAttachmentCount; ++i) k.e(r->pColorAttachmentFormats[i]);
    k.e(r->depthAttachmentFormat);
    k.e(r->stencilAttachmentFormat);
    auto present = [&](const void* p) { k.raw(uint8_t(p ? 1 : 0)); };
    present(ci.pVertexInputState);
    if (const auto* v = ci.pVertexInputState)
    {
        k.none(v->pNext);
        k.f(v->flags);
        k.raw(v->vertexBindingDescriptionCount);
        for (uint32_t i = 0; i < v->vertexBindingDescriptionCount; ++i)
        {
            k.raw(v->pVertexBindingDescriptions[i].binding);
            k.raw(v->pVertexBindingDescriptions[i].stride);
            k.e(v->pVertexBindingDescriptions[i].inputRate);
        }
        k.raw(v->vertexAttributeDescriptionCount);
        for (uint32_t i = 0; i < v->vertexAttributeDescriptionCount; ++i)
        {
            k.raw(v->pVertexAttributeDescriptions[i].location);
            k.raw(v->pVertexAttributeDescriptions[i].binding);
            k.e(v->pVertexAttributeDescriptions[i].format);
            k.raw(v->pVertexAttributeDescriptions[i].offset);
        }
    }
    present(ci.pInputAssemblyState);
    if (const auto* a = ci.pInputAssemblyState)
    {
        k.none(a->pNext);
        k.f(a->flags);
        k.e(a->topology);
        k.b(a->primitiveRestartEnable);
    }
    present(ci.pTessellationState);
    if (const auto* t = ci.pTessellationState)
    {
        k.none(t->pNext);
        k.f(t->flags);
        k.raw(t->patchControlPoints);
    }
    present(ci.pViewportState);
    if (const auto* v = ci.pViewportState)
    {
        k.none(v->pNext);
        k.f(v->flags);
        k.raw(v->viewportCount);
        present(v->pViewports);
        if (v->pViewports)
            for (uint32_t i = 0; i < v->viewportCount; ++i)
            {
                const auto& w = v->pViewports[i];
                for (float x : {w.x, w.y, w.width, w.height, w.minDepth, w.maxDepth}) k.raw(x);
            }
        k.raw(v->scissorCount);
        present(v->pScissors);
        if (v->pScissors)
            for (uint32_t i = 0; i < v->scissorCount; ++i)
            {
                const auto& s = v->pScissors[i];
                for (int32_t x : {s.offset.x, s.offset.y}) k.raw(x);
                for (uint32_t x : {s.extent.width, s.extent.height}) k.raw(x);
            }
    }
    present(ci.pRasterizationState);
    if (const auto* s = ci.pRasterizationState)
    {
        k.none(s->pNext);
        k.f(s->flags);
        k.b(s->depthClampEnable);
        k.b(s->rasterizerDiscardEnable);
        k.e(s->polygonMode);
        k.f(s->cullMode);
        k.e(s->frontFace);
        k.b(s->depthBiasEnable);
        for (float x : {s->depthBiasConstantFactor, s->depthBiasClamp, s->depthBiasSlopeFactor, s->lineWidth}) k.raw(x);
    }
    present(ci.pMultisampleState);
    if (const auto* m = ci.pMultisampleState)
    {
        k.none(m->pNext);
        k.f(m->flags);
        k.e(m->rasterizationSamples);
        k.b(m->sampleShadingEnable);
        k.raw(m->minSampleShading);
        present(m->pSampleMask);
        if (m->pSampleMask)
            for (uint32_t i = 0; i < (uint32_t(m->rasterizationSamples) + 31) / 32; ++i) k.raw(m->pSampleMask[i]);
        k.b(m->alphaToCoverageEnable);
        k.b(m->alphaToOneEnable);
    }
    present(ci.pDepthStencilState);
    if (const auto* d = ci.pDepthStencilState)
    {
        k.none(d->pNext);
        k.f(d->flags);
        k.b(d->depthTestEnable);
        k.b(d->depthWriteEnable);
        k.e(d->depthCompareOp);
        k.b(d->depthBoundsTestEnable);
        k.b(d->stencilTestEnable);
        k.stencil(d->front);
        k.stencil(d->back);
        k.raw(d->minDepthBounds);
        k.raw(d->maxDepthBounds);
    }
    present(ci.pColorBlendState);
    if (const auto* c = ci.pColorBlendState)
    {
        k.none(c->pNext);
        k.f(c->flags);
        k.b(c->logicOpEnable);
        k.e(c->logicOp);
        k.raw(c->attachmentCount);
        for (uint32_t i = 0; i < c->attachmentCount; ++i)
        {
            const auto& a = c->pAttachments[i];
            k.b(a.blendEnable);
            k.e(a.srcColorBlendFactor);
            k.e(a.dstColorBlendFactor);
            k.e(a.colorBlendOp);
            k.e(a.srcAlphaBlendFactor);
            k.e(a.dstAlphaBlendFactor);
            k.e(a.alphaBlendOp);
            k.f(a.colorWriteMask);
        }
        for (int i = 0; i < 4; ++i) k.raw(c->blendConstants[i]);
    }
    present(ci.pDynamicState);
    if (const auto* d = ci.pDynamicState)
    {
        k.none(d->pNext);
        k.f(d->flags);
        k.raw(d->dynamicStateCount);
        for (uint32_t i = 0; i < d->dynamicStateCount; ++i) k.e(d->pDynamicStates[i]);
    }
    k.raw(ci.subpass);
    if (ci.basePipelineHandle || ci.renderPass) k.valid = false;
    writeLayoutKey(k.bytes);
    return k.valid ? k.bytes : std::string();
}

vk::Pipeline Pipeline::createPipeline(vk::PipelineLayout layout)
{
    auto pipelineCreateInfo = pipelineInfo.createCreateInfo(layout, formats);
    shaderPipeline.addToPipeline(pipelineCreateInfo);
    auto result = device.createGraphicsPipeline(base->pipelineCache, pipelineCreateInfo);
    SAIGA_ASSERT(result.result == vk::Result::eSuccess && result.value);
    return result.value;
}

void Pipeline::reload()
{
    shaderPipeline.reload();
    if (!shaderPipeline.valid()) return;
    replacePipeline(createPipeline(pipelineLayout));
}

bool Pipeline::checkShader()
{
    if (autoReload && shaderPipeline.autoReload() && shaderPipeline.valid())
    {
        // The old pipeline is destroyed when the frames in flight are done with it.
        replacePipeline(createPipeline(pipelineLayout));
    }
    return pipeline;
}

}  // namespace Vulkan
}  // namespace Saiga
