/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/TerrainRenderer.h"

#include "saiga/core/image/templatedImage.h"

#include <cmath>
#include <limits>

namespace Saiga
{
namespace Vulkan
{
namespace
{
struct TerrainPush
{
    float gridN;
    float pad0, pad1;
    int wire;
};

std::vector<vec2> gridVerticesOf(int n)
{
    std::vector<vec2> v;
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) v.emplace_back(float(i), float(j));
    return v;
}

// Each cell is split into (i,j) (i,j+1) (i+1,j) and (i+1,j) (i,j+1) (i+1,j+1): the same diagonal as
// HeightField::height(), counter-clockwise seen from above.
std::vector<uint16_t> gridIndicesOf(int n)
{
    std::vector<uint16_t> idx;
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            uint16_t a = uint16_t(j * (n + 1) + i);
            uint16_t b = uint16_t(a + 1);
            uint16_t c = uint16_t(a + n + 1);
            uint16_t d = uint16_t(c + 1);
            idx.insert(idx.end(), {a, c, b, b, c, d});
        }
    }
    return idx;
}

}  // namespace

void TerrainRenderer::init(VulkanBase& _base, const RenderingFormats& formats, UniformBuffer& worldUniforms,
                           const HeightField& heightField, float _leafSize)
{
    base     = &_base;
    hf       = &heightField;
    leafSize = _leafSize;

    worldSize     = hf->worldSize();
    leavesPerSide = int(std::round(worldSize / leafSize));
    SAIGA_ASSERT(leavesPerSide > 0 && (leavesPerSide & (leavesPerSide - 1)) == 0,
                 "TerrainRenderer: the world size / leaf size must be a power of two");
    SAIGA_ASSERT(std::abs(leavesPerSide * leafSize - worldSize) < 0.01f);
    levels = 1;
    while ((1 << (levels - 1)) < leavesPerSide) levels++;

    // Min/max height of the leaves, then the higher levels
    int texelsPerLeaf = int(std::round(leafSize / hf->spacing()));
    minMax.assign(levels, {});
    minMax[0].resize(size_t(leavesPerSide) * leavesPerSide);
    for (int lz = 0; lz < leavesPerSide; ++lz)
    {
        for (int lx = 0; lx < leavesPerSide; ++lx)
        {
            float mn = std::numeric_limits<float>::max(), mx = -mn;
            for (int j = lz * texelsPerLeaf; j <= (lz + 1) * texelsPerLeaf; ++j)
            {
                for (int i = lx * texelsPerLeaf; i <= (lx + 1) * texelsPerLeaf; ++i)
                {
                    float h = hf->sample(i, j);
                    mn      = std::min(mn, h);
                    mx      = std::max(mx, h);
                }
            }
            minMax[0][size_t(lz) * leavesPerSide + lx] = vec2(mn, mx);
        }
    }
    for (int l = 1; l < levels; ++l)
    {
        int n     = leavesPerSide >> l;
        int nPrev = n * 2;
        minMax[l].resize(size_t(n) * n);
        for (int z = 0; z < n; ++z)
        {
            for (int x = 0; x < n; ++x)
            {
                vec2 r(std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
                for (int c = 0; c < 4; ++c)
                {
                    vec2 m = minMax[l - 1][size_t(2 * z + c / 2) * nPrev + 2 * x + c % 2];
                    r(0)   = std::min(r(0), m(0));
                    r(1)   = std::max(r(1), m(1));
                }
                minMax[l][size_t(z) * n + x] = r;
            }
        }
    }

    // Height map texture (row j = z)
    {
        TemplatedImage<float> img(hf->size(), hf->size());
        for (int j = 0; j < hf->size(); ++j)
            for (int i = 0; i < hf->size(); ++i) img(j, i) = hf->sample(i, j);
        heightMap.fromImage(*base, img, vk::ImageUsageFlagBits::eSampled, false, false);
    }

    // The grid meshes: full resolution, then half resolution
    {
        auto full           = gridVerticesOf(gridResolution);
        auto quarter        = gridVerticesOf(gridResolution / 2);
        quarterVertexOffset = uint32_t(full.size());
        full.insert(full.end(), quarter.begin(), quarter.end());
        gridVertices.createBuffer(*base, full.size() * sizeof(vec2),
                                  vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal);
        gridVertices.stagedUpload(*base, full.size() * sizeof(vec2), full.data());
        fullIndices.init(*base, gridIndicesOf(gridResolution), vk::MemoryPropertyFlagBits::eDeviceLocal);
        quarterIndices.init(*base, gridIndicesOf(gridResolution / 2), vk::MemoryPropertyFlagBits::eDeviceLocal);
    }

    instanceBuffer.createBuffer(*base, maxInstances * sizeof(NodeInstance),
                                vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                vk::MemoryPropertyFlagBits::eDeviceLocal);

    DescriptorSetLayout layout({
        {0,
         {0, vk::DescriptorType::eUniformBuffer, 1,
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment}},
        {1, {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eVertex}},
    });

    PipelineInfo info = worldPipelineInfo();
    info.addVertexBinding(0, sizeof(vec2), vk::VertexInputRate::eVertex, {{0, 0, vk::Format::eR32G32Sfloat, 0}});
    info.addVertexBinding(1, sizeof(NodeInstance), vk::VertexInputRate::eInstance,
                          {{1, 1, vk::Format::eR32G32B32A32Sfloat, 0}});

    std::vector<std::string> shaders = {"vulkan/world/terrain.vert", "vulkan/world/terrain.frag"};
    createWorldPipeline(pipeline, *base, layout, sizeof(TerrainPush), shaders, formats, info);
    if (wireframeSupported(*base))
        createWorldPipeline(wirePipeline, *base, layout, sizeof(TerrainPush), shaders, formats, wireframeVariant(info));

    descriptorSet = pipeline.createDescriptorSet();
    descriptorSet.assign(0, &worldUniforms);
    descriptorSet.assign(1, &heightMap);
}

void TerrainRenderer::destroy()
{
    pipeline.destroy();
    wirePipeline.destroy();
    heightMap.destroy();
    gridVertices.destroy();
    fullIndices.destroy();
    quarterIndices.destroy();
    instanceBuffer.destroy();
}

void TerrainRenderer::nodeBounds(int level, int x, int z, vec3& bmin, vec3& bmax) const
{
    float size = leafSize * float(1 << level);
    vec2 mm    = minMax[level][size_t(z) * (leavesPerSide >> level) + x];
    bmin       = vec3(x * size, mm(0), z * size);
    bmax       = vec3((x + 1) * size, mm(1), (z + 1) * size);
}

bool TerrainRenderer::select(int level, int x, int z, const vec3& cameraPos, const std::array<vec4, 6>& frustum,
                             const std::vector<float>& ranges)
{
    vec3 bmin, bmax;
    nodeBounds(level, x, z, bmin, bmax);
    float d2 = boxDistanceSquared(cameraPos, bmin, bmax);
    if (d2 > ranges[level] * ranges[level]) return false;
    if (!boxInFrustum(frustum, bmin, bmax)) return true;

    float size = leafSize * float(1 << level);
    if (level == 0 || d2 > ranges[level - 1] * ranges[level - 1])
    {
        fullNodes.push_back({vec4(x * size, z * size, size, float(level))});
        statistics.nodesPerLevel[level]++;
        statistics.trianglesPerLevel[level] += gridResolution * gridResolution * 2;
        return true;
    }

    float childSize = size * 0.5f;
    for (int c = 0; c < 4; ++c)
    {
        int cx = 2 * x + c % 2, cz = 2 * z + c / 2;
        if (!select(level - 1, cx, cz, cameraPos, frustum, ranges))
        {
            // The child is out of range of its level: draw its area with the spacing of this level.
            vec3 cmin, cmax;
            nodeBounds(level - 1, cx, cz, cmin, cmax);
            if (boxInFrustum(frustum, cmin, cmax))
            {
                quarterNodes.push_back({vec4(cx * childSize, cz * childSize, childSize, float(level))});
                statistics.nodesPerLevel[level]++;
                statistics.trianglesPerLevel[level] += gridResolution * gridResolution / 2;
            }
        }
    }
    return true;
}

void TerrainRenderer::update(vk::CommandBuffer cmd, const vec3& cameraPos, const std::array<vec4, 6>& frustum,
                             float range0, bool water)
{
    // A node of level l must fit into the morph area of level l + 1, see the CDLOD paper. With the morph starting at
    // 70% of the range this holds for range0 >= ~2.5 node diagonals.
    float minRange0 = 2.5f * leafSize * 1.42f;
    usedRange0      = std::max(range0, minRange0);

    std::vector<float> ranges(levels);
    for (int l = 0; l < levels; ++l) ranges[l] = usedRange0 * float(1 << l);
    ranges[levels - 1] = std::numeric_limits<float>::max();

    fullNodes.clear();
    quarterNodes.clear();
    waterNodes.clear();
    statistics = Stats();
    select(levels - 1, 0, 0, cameraPos, frustum, ranges);

    if (water)
    {
        // One large plane around the camera, to the horizon
        float size = 400000.0f;
        float snap = 1000.0f;
        vec2 c(std::round(cameraPos(0) / snap) * snap, std::round(cameraPos(2) / snap) * snap);
        waterNodes.push_back({vec4(c(0) - size * 0.5f, c(1) - size * 0.5f, size, -1.0f)});
    }

    std::vector<NodeInstance> all;
    all.reserve(fullNodes.size() + quarterNodes.size() + waterNodes.size());
    all.insert(all.end(), fullNodes.begin(), fullNodes.end());
    all.insert(all.end(), quarterNodes.begin(), quarterNodes.end());
    all.insert(all.end(), waterNodes.begin(), waterNodes.end());
    if ((int)all.size() > maxInstances)
    {
        // Should not happen with sensible ranges
        all.resize(maxInstances);
        fullNodes.resize(std::min<size_t>(fullNodes.size(), maxInstances));
        quarterNodes.resize(std::min<size_t>(quarterNodes.size(), maxInstances - fullNodes.size()));
        waterNodes.clear();
    }
    if (!all.empty()) instanceBuffer.update(cmd, all.size() * sizeof(NodeInstance), all.data());

    statistics.nodes        = int(fullNodes.size());
    statistics.quarterNodes = int(quarterNodes.size());
    statistics.waterNodes   = int(waterNodes.size());
    statistics.triangles    = int(fullNodes.size()) * gridResolution * gridResolution * 2 +
                           int(quarterNodes.size() + waterNodes.size()) * gridResolution * gridResolution / 2;
}

size_t TerrainRenderer::memoryBytes() const
{
    size_t heightMapBytes = size_t(heightMap.width) * heightMap.height * sizeof(float);
    return heightMapBytes + gridVertices.size() + fullIndices.size() + quarterIndices.size() + instanceBuffer.size();
}

vec2 TerrainRenderer::heightRange(float x0, float z0, float x1, float z1) const
{
    int lx0 = std::max(0, int(std::floor(x0 / leafSize)));
    int lz0 = std::max(0, int(std::floor(z0 / leafSize)));
    int lx1 = std::min(leavesPerSide - 1, int(std::floor(x1 / leafSize)));
    int lz1 = std::min(leavesPerSide - 1, int(std::floor(z1 / leafSize)));
    vec2 r(std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    for (int z = lz0; z <= lz1; ++z)
    {
        for (int x = lx0; x <= lx1; ++x)
        {
            vec2 m = minMax[0][size_t(z) * leavesPerSide + x];
            r(0)   = std::min(r(0), m(0));
            r(1)   = std::max(r(1), m(1));
        }
    }
    if (r(0) > r(1)) r = vec2(hf->waterLevel(), hf->waterLevel());
    return r;
}

void TerrainRenderer::render(vk::CommandBuffer cmd, bool wireframe, bool terrain)
{
    Pipeline& p = wireframe ? wirePipeline : pipeline;
    if (!p.bind(cmd)) return;
    p.bindDescriptorSet(cmd, descriptorSet);

    vk::Buffer grid      = gridVertices.memoryLocation()->data;
    vk::Buffer instances = instanceBuffer.memoryLocation()->data;
    cmd.bindVertexBuffers(0, grid, vk::DeviceSize(0));
    cmd.bindVertexBuffers(1, instances, vk::DeviceSize(0));

    TerrainPush push = {float(gridResolution), 0, 0, wireframe ? 1 : 0};
    auto stages      = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;

    // The instances are stored in the order full nodes, quarter nodes, water nodes
    uint32_t numFull = uint32_t(fullNodes.size());
    if (numFull > 0 && terrain)
    {
        p.pushConstant(cmd, stages, sizeof(push), &push);
        fullIndices.bind(cmd);
        cmd.drawIndexed(fullIndices.indexCount, numFull, 0, 0, 0);
    }

    uint32_t numWater   = wireframe ? 0 : uint32_t(waterNodes.size());
    uint32_t first      = terrain ? numFull : numFull + uint32_t(quarterNodes.size());
    uint32_t numQuarter = (terrain ? uint32_t(quarterNodes.size()) : 0) + numWater;
    if (numQuarter > 0)
    {
        push.gridN = float(gridResolution / 2);
        p.pushConstant(cmd, stages, sizeof(push), &push);
        quarterIndices.bind(cmd);
        cmd.drawIndexed(quarterIndices.indexCount, numQuarter, 0, int32_t(quarterVertexOffset), first);
    }
}

}  // namespace Vulkan
}  // namespace Saiga
