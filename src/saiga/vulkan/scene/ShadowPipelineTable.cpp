/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/ShadowPipelineTable.h"

#include "saiga/vulkan/scene/SceneCulling.h"
#include "saiga/vulkan/scene/SceneLightingSet.h"

#include <tuple>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
namespace
{
// shader/vulkan/scene/lighting/shadow_depth.vert, shadow_foliage.vert: Push
struct ShadowPush
{
    mat4 viewProj;
    float alpha;
    float pad[3];
};
static_assert(sizeof(ShadowPush) == 80, "Push of the shadow shaders");
}  // namespace

ShadowPipelineTable::Entry ShadowPipelineTable::defaultEntry(int klass)
{
    // MeshPool::pipelineClassIndex: opaque, alpha-masked, two-sided, alpha-masked two-sided, prototype foliage
    Entry e;
    e.vertexShader = "vulkan/scene/lighting/shadow_depth.vert";
    e.twoSided     = klass >= 2;
    if (klass == 1 || klass == 3) e.fragmentShader = "vulkan/scene/lighting/shadow_alpha.frag";
    if (klass == int(PipelineClass::PrototypeFoliage) - 1)
    {
        e.vertexShader   = "vulkan/scene/lighting/shadow_foliage.vert";
        e.fragmentShader = "vulkan/scene/lighting/shadow_foliage.frag";
    }
    return e;
}

void ShadowPipelineTable::init(VulkanBase& b)
{
    destroy();
    base = &b;
    for (int c = 0; c < pipelineClassCount; ++c) setEntry(c, defaultEntry(c));
}

void ShadowPipelineTable::destroy()
{
    if (!base) return;
    for (auto& p : pipelines)
        if (p) p->destroy();
    for (auto& p : pipelines) p.reset();
    for (auto& e : entries) e.reset();
    base = nullptr;
}

void ShadowPipelineTable::setEntry(int klass, std::optional<Entry> entry)
{
    SAIGA_ASSERT(klass >= 0 && klass < pipelineClassCount, "ShadowPipelineTable::setEntry: no such class");
    // Pipeline::destroy releases the pipeline after the frames in flight
    if (pipelines[size_t(klass)]) pipelines[size_t(klass)]->destroy();
    pipelines[size_t(klass)].reset();
    entries[size_t(klass)] = std::move(entry);
    if (entries[size_t(klass)]) create(klass);
}

void ShadowPipelineTable::create(int klass)
{
    const Entry& e = *entries[size_t(klass)];
    auto p         = std::make_unique<Pipeline>();
    PipelineInfo info;
    info.depthStencilState.depthCompareOp   = vk::CompareOp::eLessOrEqual;
    info.depthStencilState.depthWriteEnable = VK_TRUE;
    info.rasterizationState.cullMode        = e.twoSided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack;
    // The slope bias (constant factor 0: the receivers' bias is in texels, shadows.glsl)
    info.rasterizationState.depthBiasEnable = VK_TRUE;
    info.dynamicStateEnables.push_back(vk::DynamicState::eDepthBias);
    info.addVertexBinding(0, sizeof(MeshVertex), vk::VertexInputRate::eVertex,
                          {{0, 0, vk::Format::eR32G32B32Sfloat, uint32_t(offsetof(MeshVertex, position))},
                           {1, 0, vk::Format::eR32Uint, uint32_t(offsetof(MeshVertex, normal))},
                           {2, 0, vk::Format::eR32Uint, uint32_t(offsetof(MeshVertex, color))},
                           {3, 0, vk::Format::eR32G32Sfloat, uint32_t(offsetof(MeshVertex, uv))}});
    p->init(*base, 2);
    p->addDescriptorSetLayout(SceneCulling::setLayout(), 0);
    p->addDescriptorSetLayout(SceneLightingSet::shadowLayout(), 1);
    p->addPushConstantRange(
        {vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(ShadowPush)});
    std::vector<std::tuple<std::string, vk::ShaderStageFlagBits, std::string>> modules = {
        {e.vertexShader, vk::ShaderStageFlagBits::eVertex, ""}};
    if (!e.fragmentShader.empty()) modules.emplace_back(e.fragmentShader, vk::ShaderStageFlagBits::eFragment, "");
    p->shaderPipeline.loadGLSL(base->device, modules);
    RenderingFormats formats;
    formats.depth = kDepthFormat;
    p->create(formats, info);
    pipelines[size_t(klass)] = std::move(p);
    creations++;
}

bool ShadowPipelineTable::bind(vk::CommandBuffer cmd, int klass, vk::DescriptorSet sceneSet,
                               vk::DescriptorSet shadowSet, const mat4& viewProj, float alpha, float slopeBias)
{
    Pipeline* p = pipelines[size_t(klass)].get();
    if (!p || !p->bind(cmd)) return false;
    ShadowPush push;
    push.viewProj = viewProj;
    push.alpha    = alpha;
    push.pad[0] = push.pad[1] = push.pad[2] = 0;
    p->bindRawDescriptorSet(cmd, sceneSet, 0);
    p->bindRawDescriptorSet(cmd, shadowSet, 1);
    p->pushConstant(cmd, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, sizeof(push), &push);
    cmd.setDepthBias(0.0f, 0.0f, slopeBias);
    return true;
}

}  // namespace Vulkan
}  // namespace Saiga
