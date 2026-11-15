/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/ForestRenderer.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
struct CullPush
{
    uint32_t treeCount;
    uint32_t speciesCount;
    uint32_t pad[2];
    // per species: x: bounding sphere center height, y: radius (units of the tree height)
    vec4 bounds[4];
};

struct DrawPush
{
    int pad[3];
    int wire;
};

float smoothstepf(float e0, float e1, float x)
{
    float t = std::min(std::max((x - e0) / (e1 - e0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

void memoryBarrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
{
    vk::MemoryBarrier2 b(srcStage, srcAccess, dstStage, dstAccess);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, b, nullptr, nullptr));
}

}  // namespace

std::vector<ForestRenderer::TreeInstance> ForestRenderer::placeTrees(const HeightField& hf, int count, uint32_t seed,
                                                                     const std::vector<TreeSpecies>& species)
{
    const float cell     = 4.0f;
    const float W        = hf.worldSize();
    const int cells      = int(W / cell);
    const float water    = hf.waterLevel();
    const float treeline = hf.parameters().mountainHeight * 0.72f;

    // Probability of a tree per cell
    std::vector<float> p(size_t(cells) * cells);
#pragma omp parallel for schedule(dynamic, 16)
    for (int j = 0; j < cells; ++j)
    {
        for (int i = 0; i < cells; ++i)
        {
            float x = (i + 0.5f) * cell, z = (j + 0.5f) * cell;
            float h = hf.height(x, z);
            vec3 n  = hf.normal(x, z, 3.0f);
            float forest =
                smoothstepf(-0.05f, 0.3f, WorldNoise::fbm(x / 700.0f, z / 700.0f, 4, seed * 31 + 1)) * 0.95f + 0.02f;
            float slope  = smoothstepf(0.8f, 0.9f, n(1));
            float height = smoothstepf(water + 1.0f, water + 4.0f, h) * smoothstepf(treeline, treeline - 80.0f, h);
            float v      = forest * slope * height;
            p[size_t(j) * cells + i] = v;
        }
    }
    // Serial sum: a parallel reduction would make the result (and the placement) depend on the scheduling
    double sum = 0;
    for (float v : p) sum += v;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0, 1);
    float scale = sum > 0 ? float(count / sum) : 0;

    std::vector<TreeInstance> trees;
    trees.reserve(size_t(count * 1.1f));
    for (int j = 0; j < cells; ++j)
    {
        for (int i = 0; i < cells; ++i)
        {
            float prob = p[size_t(j) * cells + i] * scale;
            if (u01(rng) >= prob) continue;
            float x = (i + u01(rng)) * cell, z = (j + u01(rng)) * cell;
            float h = hf.height(x, z);

            // Spruce in the mountains, broadleaf and birch in the valleys (in patches)
            float spruce = smoothstepf(70, 200, h) * 0.85f +
                           0.15f * smoothstepf(-0.2f, 0.4f, WorldNoise::fbm(x / 400.0f, z / 400.0f, 2, seed + 7));
            int s;
            if (u01(rng) < spruce)
                s = 0;
            else
                s = WorldNoise::fbm(x / 300.0f, z / 300.0f, 2, seed + 9) + u01(rng) * 0.6f - 0.3f > 0 ? 1 : 2;
            s = std::min(s, int(species.size()) - 1);

            vec2 hr = species[s].heightRange;
            TreeInstance t;
            // Sink the stem base a little into slopes
            t.posYaw       = vec4(x, h - 0.25f, z, u01(rng) * 6.2831853f);
            t.scaleSpecies = vec4(hr(0) + (hr(1) - hr(0)) * u01(rng), float(s), 0, 0);
            trees.push_back(t);
        }
    }
    if ((int)trees.size() > count)
    {
        std::shuffle(trees.begin(), trees.end(), rng);
        trees.resize(count);
    }
    return trees;
}

void ForestRenderer::init(VulkanBase& _base, const RenderingFormats& formats, UniformBuffer& worldUniforms,
                          Texture2DArray& foliageTexture, const HeightField& heightField, int treeCount, uint32_t seed)
{
    base    = &_base;
    species = FoliageGenerator::treeSpecies(seed);
    SAIGA_ASSERT(species.size() <= 4);
    instances = placeTrees(heightField, treeCount, seed, species);
    SAIGA_ASSERT(instances.size() < (1u << 30), "ForestRenderer: too many trees");
    SAIGA_ASSERT(base->enabledFeatures.drawIndirectFirstInstance,
                 "ForestRenderer: the drawIndirectFirstInstance feature is required");

    // Sort the instances by species: better locality
    std::stable_sort(instances.begin(), instances.end(), [](const TreeInstance& a, const TreeInstance& b)
                     { return a.scaleSpecies(1) < b.scaleSpecies(1); });

    std::vector<int> perSpecies(species.size(), 0);
    for (auto& t : instances) perSpecies[int(t.scaleSpecies(1))]++;

    // Meshes and draw commands, bucket = species * lodCount + lod
    std::vector<TreeVertex> vertices;
    std::vector<uint32_t> indices;
    uint32_t firstInstance = 0;
    commandTemplate.clear();
    bucketTriangles.clear();
    for (size_t s = 0; s < species.size(); ++s)
    {
        for (int l = 0; l < lodCount; ++l)
        {
            auto& m = species[s].lods[l];
            vk::DrawIndexedIndirectCommand c;
            c.indexCount    = uint32_t(m.indices.size());
            c.instanceCount = 0;
            c.firstIndex    = uint32_t(indices.size());
            c.vertexOffset  = int32_t(vertices.size());
            c.firstInstance = firstInstance;
            commandTemplate.push_back(c);
            bucketTriangles.push_back(m.triangles());

            vertices.insert(vertices.end(), m.vertices.begin(), m.vertices.end());
            indices.insert(indices.end(), m.indices.begin(), m.indices.end());
            firstInstance += uint32_t(perSpecies[s]);
        }
    }
    uint32_t visibleCapacity = std::max(firstInstance, 1u);

    auto deviceLocal = vk::MemoryPropertyFlagBits::eDeviceLocal;
    vertexBuffer.createBuffer(*base, vertices.size() * sizeof(TreeVertex),
                              vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                              deviceLocal);
    vertexBuffer.stagedUpload(*base, vertices.size() * sizeof(TreeVertex), vertices.data());
    indexBuffer.createBuffer(*base, indices.size() * sizeof(uint32_t),
                             vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                             deviceLocal);
    indexBuffer.stagedUpload(*base, indices.size() * sizeof(uint32_t), indices.data());

    size_t instanceBytes = std::max<size_t>(instances.size(), 1) * sizeof(TreeInstance);
    instanceBuffer.createBuffer(*base, instanceBytes,
                                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                deviceLocal);
    if (!instances.empty())
        instanceBuffer.stagedUpload(*base, instances.size() * sizeof(TreeInstance), instances.data());

    visibleBuffer.createBuffer(*base, visibleCapacity * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer,
                               deviceLocal);

    size_t commandBytes = commandTemplate.size() * sizeof(vk::DrawIndexedIndirectCommand);
    commandBuffer.createBuffer(*base, commandBytes,
                               vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
                                   vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                               deviceLocal);
    commandBuffer.stagedUpload(*base, commandBytes, commandTemplate.data());
    for (auto& r : readbacks)
    {
        r.buffer.createBuffer(*base, commandBytes, vk::BufferUsageFlagBits::eTransferDst,
                              vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        r.frameValue = 0;
    }

    // Culling
    {
        DescriptorSetLayout layout({
            {0, {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {1, {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {2, {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {3, {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
        });
        cullPipeline.init(*base, 1);
        cullPipeline.addDescriptorSetLayout(layout);
        cullPipeline.addPushConstantRange({vk::ShaderStageFlagBits::eCompute, 0, sizeof(CullPush)});
        cullPipeline.shaderPipeline.loadCompute(base->device, "vulkan/world/tree_cull.comp");
        cullPipeline.create();
        cullSet = cullPipeline.createDescriptorSet();
        cullSet.assign(0, &worldUniforms);
        cullSet.assign(1, &instanceBuffer);
        cullSet.assign(2, &visibleBuffer);
        cullSet.assign(3, &commandBuffer);
    }

    // Drawing
    {
        auto vf = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
        DescriptorSetLayout layout({
            {0, {0, vk::DescriptorType::eUniformBuffer, 1, vf}},
            {1, {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}},
            {2, {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}},
            {3, {3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}},
        });
        PipelineInfo info                = worldPipelineInfo();
        info.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        // The leaf cards: alpha to coverage (with MSAA), the opaque parts write alpha = 1
        info.multisampleState.alphaToCoverageEnable = VK_TRUE;
        info.addVertexBinding(0, sizeof(TreeVertex), vk::VertexInputRate::eVertex,
                              {{0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(TreeVertex, posAo)},
                               {1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(TreeVertex, normalWind)},
                               {2, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(TreeVertex, uvLayer)}});
        std::vector<std::string> shaders = {"vulkan/world/tree.vert", "vulkan/world/foliage.frag"};
        createWorldPipeline(pipeline, *base, layout, sizeof(DrawPush), shaders, formats, info);
        if (wireframeSupported(*base))
            createWorldPipeline(wirePipeline, *base, layout, sizeof(DrawPush), shaders, formats,
                                wireframeVariant(info));
        drawSet = pipeline.createDescriptorSet();
        drawSet.assign(0, &worldUniforms);
        drawSet.assign(1, &instanceBuffer);
        drawSet.assign(2, &visibleBuffer);
        drawSet.assign(3, &foliageTexture);
    }
}

void ForestRenderer::destroy()
{
    cullPipeline.destroy();
    pipeline.destroy();
    wirePipeline.destroy();
    vertexBuffer.destroy();
    indexBuffer.destroy();
    instanceBuffer.destroy();
    visibleBuffer.destroy();
    commandBuffer.destroy();
    for (auto& r : readbacks) r.buffer.destroy();
}

size_t ForestRenderer::memoryBytes() const
{
    size_t bytes =
        vertexBuffer.size() + indexBuffer.size() + instanceBuffer.size() + visibleBuffer.size() + commandBuffer.size();
    for (auto& r : readbacks) bytes += r.buffer.size();
    return bytes;
}

void ForestRenderer::readStats()
{
    // The newest readback of a completed frame
    uint64_t completed = base->completedFrameValue();
    Readback* best     = nullptr;
    for (auto& r : readbacks)
    {
        if (r.frameValue > 0 && r.frameValue <= completed && (!best || r.frameValue > best->frameValue)) best = &r;
    }
    if (!best) return;

    std::vector<vk::DrawIndexedIndirectCommand> cmds(commandTemplate.size());
    best->buffer.download(cmds.data(), cmds.size() * sizeof(vk::DrawIndexedIndirectCommand));
    statistics = Stats();
    statistics.bucketVisible.assign(cmds.size(), 0);
    for (size_t b = 0; b < cmds.size(); ++b)
    {
        statistics.bucketVisible[b] = int(cmds[b].instanceCount);
        statistics.visible[b % lodCount] += int(cmds[b].instanceCount);
        statistics.triangles += int(cmds[b].instanceCount) * bucketTriangles[b];
    }
}

void ForestRenderer::cull(vk::CommandBuffer cmd)
{
    if (instances.empty()) return;
    readStats();

    // The previous frames read (draw) and wrote (cull) the commands and the visible list.
    memoryBarrier(
        cmd, Stage::eDrawIndirect | Stage::eVertexShader | Stage::eComputeShader, Access::eShaderStorageWrite,
        Stage::eAllTransfer | Stage::eComputeShader,
        Access::eTransferRead | Access::eTransferWrite | Access::eShaderStorageRead | Access::eShaderStorageWrite);

    // Statistics: copy the result of the previous frame
    {
        Readback& r          = readbacks[readbackIndex];
        readbackIndex        = (readbackIndex + 1) % readbacks.size();
        vk::DeviceSize bytes = commandTemplate.size() * sizeof(vk::DrawIndexedIndirectCommand);
        commandBuffer.copyTo(cmd, r.buffer, 0, 0, bytes);
        r.frameValue = base->currentFrameSignalValue();
        // The reset below overwrites the commands the copy reads (write after read)
        memoryBarrier(cmd, Stage::eAllTransfer, Access::eTransferRead, Stage::eAllTransfer, Access::eTransferWrite);
    }

    // Reset the instance counts
    commandBuffer.update(cmd, commandTemplate.size() * sizeof(vk::DrawIndexedIndirectCommand), commandTemplate.data());
    memoryBarrier(cmd, Stage::eAllTransfer, Access::eTransferWrite, Stage::eComputeShader | Stage::eHost,
                  Access::eShaderStorageRead | Access::eShaderStorageWrite | Access::eHostRead);

    if (cullPipeline.bind(cmd))
    {
        CullPush push     = {};
        push.treeCount    = uint32_t(instances.size());
        push.speciesCount = uint32_t(species.size());
        for (size_t s = 0; s < species.size(); ++s)
        {
            push.bounds[s] = vec4(species[s].bounds(0), species[s].bounds(1), 0, 0);
        }
        cullPipeline.bindDescriptorSet(cmd, cullSet);
        cullPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        cmd.dispatch((push.treeCount + 255) / 256, 1, 1);
    }

    memoryBarrier(cmd, Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eDrawIndirect | Stage::eVertexShader,
                  Access::eIndirectCommandRead | Access::eShaderStorageRead);
}

int ForestRenderer::drawCalls() const
{
    if (instances.empty()) return 0;
    return base->enabledFeatures.multiDrawIndirect ? 1 : drawCommands();
}

void ForestRenderer::render(vk::CommandBuffer cmd, bool wireframe)
{
    if (instances.empty()) return;
    Pipeline& p = wireframe ? wirePipeline : pipeline;
    if (!p.bind(cmd)) return;
    p.bindDescriptorSet(cmd, drawSet);

    DrawPush push = {{0, 0, 0}, wireframe ? 1 : 0};
    p.pushConstant(cmd, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, sizeof(push), &push);

    cmd.bindVertexBuffers(0, vk::Buffer(vertexBuffer.memoryLocation()->data), vk::DeviceSize(0));
    cmd.bindIndexBuffer(indexBuffer.memoryLocation()->data, 0, vk::IndexType::eUint32);

    vk::Buffer commands = commandBuffer.memoryLocation()->data;
    uint32_t count      = uint32_t(commandTemplate.size());
    uint32_t stride     = sizeof(vk::DrawIndexedIndirectCommand);
    if (base->enabledFeatures.multiDrawIndirect)
    {
        cmd.drawIndexedIndirect(commands, 0, count, stride);
    }
    else
    {
        for (uint32_t i = 0; i < count; ++i) cmd.drawIndexedIndirect(commands, i * stride, 1, stride);
    }
}

}  // namespace Vulkan
}  // namespace Saiga
