/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/GrassPass.h"

#include "saiga/core/math/ChunkId.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/SceneCulling.h"
#include "saiga/vulkan/scene/SceneLightingSet.h"
#include "saiga/vulkan/scene/TerrainAtlas.h"
#include "saiga/vulkan/scene/TerrainPass.h"
#include "saiga/vulkan/world/FoliageGenerator.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Vulkan
{
namespace
{
static_assert(sizeof(GrassPass::BlockDraw) == 32, "the push constants of grass.vert");

vk::Buffer handle(Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}

bool boxCulled(const Vec3& lo, const Vec3& hi, const std::array<Vec4, 5>& planes)
{
    for (const Vec4& p : planes)
    {
        const Vec3 c(p(0) >= 0 ? hi(0) : lo(0), p(1) >= 0 ? hi(1) : lo(1), p(2) >= 0 ? hi(2) : lo(2));
        if (p(0) * c(0) + p(1) * c(1) + p(2) * c(2) + p(3) < 0) return true;
    }
    return false;
}

// The clumps are at most this high (grass.vert)
constexpr double maxClumpHeight = 1.6;
}  // namespace

int GrassPass::cellsPerChunk(double cellSize)
{
    const double c = std::max(cellSize, 0.05);
    return 16 * std::max(1, int(std::lround(256.0 / (16.0 * c))));
}

void GrassPass::init(VulkanBase& b, const RenderingFormats& formats, SceneLightingSet& lighting, bool pbr)
{
    destroy();
    base             = &b;
    lightingSet      = &lighting;
    sceneFormats     = formats;
    FoliageMesh mesh = FoliageGenerator::grassClump();
    indexCount       = uint32_t(mesh.indices.size());
    vertices.createBuffer(*base, mesh.vertices.size() * sizeof(TreeVertex),
                          vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                          vk::MemoryPropertyFlagBits::eDeviceLocal);
    vertices.stagedUpload(*base, mesh.vertices.size() * sizeof(TreeVertex), mesh.vertices.data());
    indices.createBuffer(*base, mesh.indices.size() * sizeof(uint32_t),
                         vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                         vk::MemoryPropertyFlagBits::eDeviceLocal);
    indices.stagedUpload(*base, mesh.indices.size() * sizeof(uint32_t), mesh.indices.data());

    pbrVariant = !pbr;
    setPbr(pbr);
    initialized = true;
}

void GrassPass::setPbr(bool pbr)
{
    if (pbr == pbrVariant) return;
    pbrVariant = pbr;
    pipeline.destroy();
    PipelineInfo info;
    info.depthStencilState.depthCompareOp       = vk::CompareOp::eGreaterOrEqual;  // reversed-Z
    info.rasterizationState.cullMode            = vk::CullModeFlagBits::eNone;
    info.multisampleState.alphaToCoverageEnable = VK_TRUE;
    info.addVertexBinding(0, sizeof(TreeVertex), vk::VertexInputRate::eVertex,
                          {{0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(TreeVertex, posAo)},
                           {1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(TreeVertex, normalWind)},
                           {2, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(TreeVertex, uvLayer)}});
    pipeline.init(*base, 2);
    pipeline.addDescriptorSetLayout(TerrainPass::setLayout(), 0);
    pipeline.addDescriptorSetLayout(SceneLightingSet::layout(), 1);
    pipeline.addPushConstantRange(
        {vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(BlockDraw)});
    const std::string defines = pbr ? "#define LIGHTING_PBR\n" : "";
    pipeline.shaderPipeline.loadGLSL(base->device,
                                     {{"vulkan/scene/grass.vert", vk::ShaderStageFlagBits::eVertex, defines},
                                      {"vulkan/scene/grass.frag", vk::ShaderStageFlagBits::eFragment, defines}});
    pipeline.create(sceneFormats, info);
}

void GrassPass::destroy()
{
    if (!initialized) return;
    pipeline.destroy();
    vertices.destroy();
    indices.destroy();
    draws.clear();
    initialized = false;
}

void GrassPass::update(const TerrainAtlas& atlas, const TerrainPass& terrain, const SceneCamera& camera,
                       const PrototypeSettings& settings)
{
    draws.clear();
    statistics = Stats();
    if (!terrain.drawsThisFrame() || !settings.grass || !(settings.grassRadius > 0)) return;  // also NaN
    const TerrainLayout& L   = atlas.layout();
    const int N              = cellsPerChunk(settings.grassCellSize);
    const ivec4 anchor       = terrain.anchor();
    const Vec3& c            = camera.position;
    const double r           = settings.grassRadius;
    statistics.cellsPerChunk = N;
    // The chunk range, clamped in double before the conversion (any finite camera; TerrainPass skips non-finite ones)
    auto chunkOf = [&](double v)
    { return int32_t(std::min(std::max(std::floor(v / 256), -1.0), double(L.sizeChunks))); };
    const int32_t cx0 = chunkOf(c.x() - r), cx1 = chunkOf(c.x() + r);
    const int32_t cz0 = chunkOf(c.z() - r), cz1 = chunkOf(c.z() + r);
    const int nodeSize = 32 * L.baseSpacing;
    for (int32_t cz = std::max(0, cz0); cz <= std::min(L.sizeChunks - 1, cz1); ++cz)
    {
        for (int32_t cx = std::max(0, cx0); cx <= std::min(L.sizeChunks - 1, cx1); ++cx)
        {
            const int layer = atlas.maskLayer(cx, cz);
            if (layer < 0 || !atlas.tileResident(0, cx, cz)) continue;
            bool any = false;
            for (int bz = 0; bz < 16; ++bz)
            {
                for (int bx = 0; bx < 16; ++bx)
                {
                    const int32_t ox = cx * 256 + bx * 16, oz = cz * 256 + bz * 16;
                    const double dx = std::max({double(ox) - c.x(), 0.0, c.x() - double(ox + 16)});
                    const double dz = std::max({double(oz) - c.z(), 0.0, c.z() - double(oz + 16)});
                    if (dx * dx + dz * dz > r * r) continue;
                    // The level-0 node of the block (bounds of its tile)
                    const TerrainAtlas::Bound b = atlas.ownBound(0, ox / nodeSize, oz / nodeSize);
                    if (!b.valid()) continue;
                    const Vec3 lo(double(ox) - 1 - c.x(), double(b.lo) * kHeightUnit - 1 - c.y(),
                                  double(oz) - 1 - c.z());
                    const Vec3 hi(double(ox + 16) + 1 - c.x(), double(b.hi) * kHeightUnit + maxClumpHeight - c.y(),
                                  double(oz + 16) + 1 - c.z());
                    if (boxCulled(lo, hi, camera.planes)) continue;
                    BlockDraw d;
                    d.originX   = ox - anchor(0);
                    d.originZ   = oz - anchor(1);
                    d.blockX    = bx;
                    d.blockZ    = bz;
                    d.chunkX    = cx;
                    d.chunkZ    = cz;
                    d.maskLayer = layer;
                    d.cells     = N;
                    draws.push_back(d);
                    any = true;
                }
            }
            statistics.chunks += any ? 1 : 0;
        }
    }
    const int64_t perBlock = int64_t(N / 16) * int64_t(N / 16);
    statistics.blocks      = int(draws.size());
    statistics.drawCalls   = int(draws.size());
    statistics.clumps      = perBlock * int64_t(draws.size());
    statistics.triangles   = statistics.clumps * int64_t(indexCount / 3);
}

void GrassPass::render(vk::CommandBuffer cmd, uint32_t frameIndex, const TerrainPass& terrain)
{
    if (draws.empty() || !terrain.drawsThisFrame()) return;
    if (!pipeline.bind(cmd)) return;
    vk::DescriptorSet set = terrain.descriptorSet(frameIndex);
    pipeline.bindRawDescriptorSet(cmd, set);
    pipeline.bindRawDescriptorSet(cmd, lightingSet->set(frameIndex), 1);
    cmd.bindVertexBuffers(0, handle(vertices), vk::DeviceSize(0));
    cmd.bindIndexBuffer(handle(indices), 0, vk::IndexType::eUint32);
    const auto stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    for (const BlockDraw& d : draws)
    {
        pipeline.pushConstant(cmd, stages, sizeof(d), &d);
        cmd.drawIndexed(indexCount, uint32_t((d.cells / 16) * (d.cells / 16)), 0, 0, 0);
    }
}

}  // namespace Vulkan
}  // namespace Saiga
