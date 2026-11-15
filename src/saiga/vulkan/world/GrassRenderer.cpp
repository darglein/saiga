/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/GrassRenderer.h"

#include "saiga/vulkan/world/FoliageGenerator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace Saiga
{
namespace Vulkan
{
namespace
{
struct GrassPush
{
    ivec2 tileOrigin;
    int tileCells;
    int wire;
};
}  // namespace

void GrassRenderer::init(VulkanBase& _base, const RenderingFormats& formats, UniformBuffer& worldUniforms,
                         Texture2DArray& foliageTexture, TerrainRenderer& _terrain)
{
    base    = &_base;
    terrain = &_terrain;

    FoliageMesh clump = FoliageGenerator::grassClump();
    indexCount        = uint32_t(clump.indices.size());
    vertexBuffer.createBuffer(*base, clump.vertices.size() * sizeof(TreeVertex),
                              vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                              vk::MemoryPropertyFlagBits::eDeviceLocal);
    vertexBuffer.stagedUpload(*base, clump.vertices.size() * sizeof(TreeVertex), clump.vertices.data());
    indexBuffer.createBuffer(*base, clump.indices.size() * sizeof(uint32_t),
                             vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                             vk::MemoryPropertyFlagBits::eDeviceLocal);
    indexBuffer.stagedUpload(*base, clump.indices.size() * sizeof(uint32_t), clump.indices.data());

    auto vf = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    DescriptorSetLayout layout({
        {0, {0, vk::DescriptorType::eUniformBuffer, 1, vf}},
        {1, {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eVertex}},
        {3, {3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}},
    });
    PipelineInfo info                           = worldPipelineInfo();
    info.rasterizationState.cullMode            = vk::CullModeFlagBits::eNone;
    info.multisampleState.alphaToCoverageEnable = VK_TRUE;
    info.addVertexBinding(0, sizeof(TreeVertex), vk::VertexInputRate::eVertex,
                          {{0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(TreeVertex, posAo)},
                           {1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(TreeVertex, normalWind)},
                           {2, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(TreeVertex, uvLayer)}});
    std::vector<std::string> shaders = {"vulkan/world/grass.vert", "vulkan/world/foliage.frag"};
    createWorldPipeline(pipeline, *base, layout, sizeof(GrassPush), shaders, formats, info);
    if (wireframeSupported(*base))
        createWorldPipeline(wirePipeline, *base, layout, sizeof(GrassPush), shaders, formats, wireframeVariant(info));

    set = pipeline.createDescriptorSet();
    set.assign(0, &worldUniforms);
    set.assign(1, &terrain->heightTexture());
    set.assign(3, &foliageTexture);
}

void GrassRenderer::destroy()
{
    pipeline.destroy();
    wirePipeline.destroy();
    vertexBuffer.destroy();
    indexBuffer.destroy();
}

void GrassRenderer::update(const vec3& cameraPos, const std::array<vec4, 6>& frustum, float radius, float cellSize)
{
    visibleTiles.clear();
    statistics = Stats();
    if (radius <= 0 || cellSize <= 0) return;

    // In 64 bit: a tiny cell size would overflow int. The cells per tile are capped (at most maxTileCells^2 clumps
    // per draw); with a capped grid the grass covers less than the radius.
    int64_t requestedCells = int64_t(std::ceil(2.0 * double(radius) / double(cellSize)));
    tileCells              = int(std::min<int64_t>((requestedCells + tiles - 1) / tiles, maxTileCells));
    int gridCells          = tileCells * tiles;

    ivec2 cameraCell(int(std::floor(cameraPos(0) / cellSize)), int(std::floor(cameraPos(2) / cellSize)));
    ivec2 first = cameraCell - ivec2(gridCells / 2, gridCells / 2);

    // The clumps are at most this high (see grass.vert)
    const float maxClumpHeight = 1.6f;
    for (int tz = 0; tz < tiles; ++tz)
    {
        for (int tx = 0; tx < tiles; ++tx)
        {
            ivec2 origin = first + ivec2(tx * tileCells, tz * tileCells);
            float x0 = origin(0) * cellSize, z0 = origin(1) * cellSize;
            float x1 = x0 + tileCells * cellSize, z1 = z0 + tileCells * cellSize;

            // Outside of the radius?
            float dx = std::max({x0 - cameraPos(0), 0.0f, cameraPos(0) - x1});
            float dz = std::max({z0 - cameraPos(2), 0.0f, cameraPos(2) - z1});
            if (dx * dx + dz * dz > radius * radius) continue;

            vec2 h = terrain->heightRange(x0, z0, x1, z1);
            if (!boxInFrustum(frustum, vec3(x0 - 1, h(0) - 1, z0 - 1), vec3(x1 + 1, h(1) + maxClumpHeight, z1 + 1)))
                continue;
            visibleTiles.push_back({origin});
        }
    }
    int64_t clumps       = int64_t(visibleTiles.size()) * tileCells * tileCells;
    int64_t triangles    = clumps * int64_t(indexCount / 3);
    statistics.tiles     = int(visibleTiles.size());
    statistics.tileCells = tileCells;
    statistics.clumps    = int(std::min<int64_t>(clumps, std::numeric_limits<int>::max()));
    statistics.triangles = int(std::min<int64_t>(triangles, std::numeric_limits<int>::max()));
}

void GrassRenderer::render(vk::CommandBuffer cmd, bool wireframe)
{
    if (visibleTiles.empty()) return;
    Pipeline& p = wireframe ? wirePipeline : pipeline;
    if (!p.bind(cmd)) return;
    p.bindDescriptorSet(cmd, set);
    cmd.bindVertexBuffers(0, vk::Buffer(vertexBuffer.memoryLocation()->data), vk::DeviceSize(0));
    cmd.bindIndexBuffer(indexBuffer.memoryLocation()->data, 0, vk::IndexType::eUint32);

    auto stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    for (auto& t : visibleTiles)
    {
        GrassPush push = {t.origin, tileCells, wireframe ? 1 : 0};
        p.pushConstant(cmd, stages, sizeof(push), &push);
        cmd.drawIndexed(indexCount, uint32_t(tileCells * tileCells), 0, 0, 0);
    }
}

}  // namespace Vulkan
}  // namespace Saiga
