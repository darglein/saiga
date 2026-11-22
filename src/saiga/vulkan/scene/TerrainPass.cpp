/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/TerrainPass.h"

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/SceneCulling.h"
#include "saiga/vulkan/scene/SceneLightingSet.h"
#include "saiga/vulkan/scene/TerrainAtlas.h"
#include "saiga/vulkan/texture/Texture.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace Saiga
{
namespace Vulkan
{
namespace
{
static_assert(sizeof(TerrainPass::GpuNode) == TerrainAtlas::nodeInstanceBytes, "TerrainNode of terrain_heights.glsl");
static_assert(sizeof(TerrainFrameUniforms) == 2 * 64 + 4 * 16 + 8 * 16 + 5 * 16, "TerrainFrame of terrain_common.glsl");

struct DrawPush
{
    int32_t gridN;
    int32_t pad;
};
// shader/vulkan/scene/lighting/terrain_shadow.vert
struct ShadowPush
{
    mat4 viewProj;
    int32_t gridN;
    int32_t pad[3];
};
static_assert(sizeof(ShadowPush) == 80, "Push of terrain_shadow.vert");
struct ProbeGpu
{
    uint32_t args[4];
    float position[4];
};

constexpr uint32_t flagQuarter = 1u << 8;
constexpr uint32_t flagWater   = 1u << 9;
// S36b: a node of the atlas's fixed level: no morph (terrain_vertex.glsl)
constexpr uint32_t flagFixed = 1u << 10;
// The atlas's nodes per frame (spheres of the shadowed lights)
constexpr uint32_t kMaxAtlasNodes = 16384;
// terrain_shadow.vert with ATLAS_DEPTH: Push
struct AtlasPush
{
    mat4 viewProj;
    int32_t gridN;
    int32_t pad[3];
    vec4 light;
};
static_assert(sizeof(AtlasPush) == 96, "Push of terrain_shadow.vert (ATLAS_DEPTH)");

vk::Buffer handle(Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}

// The indices of an n x n grid in the (n + 3)^2 vertex layout (the outer ring: the skirt), with the skirt cells or
// only the inner cells
std::vector<uint16_t> gridIndices(int n, bool skirt)
{
    const int ext = n + 3;
    std::vector<uint16_t> tri, out;
    if (skirt)
    {
        // Every cell of the (n + 2)^2 cells: the inner grid and the vertical walls to the skirt ring (outward facing)
        terrainGridTriangles(n + 2, tri);
        return tri;
    }
    terrainGridTriangles(n, tri);
    for (uint16_t v : tri)
    {
        const int i = v % (n + 1), j = v / (n + 1);
        out.push_back(uint16_t((j + 1) * ext + (i + 1)));
    }
    return out;
}

// The box [lo, hi] (camera-relative) is outside a frustum plane
template <size_t N>
bool boxCulled(const Vec3& lo, const Vec3& hi, const std::array<Vec4, N>& planes)
{
    for (const Vec4& p : planes)
    {
        const Vec3 c(p(0) >= 0 ? hi(0) : lo(0), p(1) >= 0 ? hi(1) : lo(1), p(2) >= 0 ? hi(2) : lo(2));
        if (p(0) * c(0) + p(1) * c(1) + p(2) * c(2) + p(3) < 0) return true;
    }
    return false;
}

// The LOD metric of the selection and the vertex morph: sqrt(dxz^2 + hc^2) with the horizontal distance dxz of the
// camera-relative box [lo, hi] (xz only) and the frame's camera height hc above the surface under the camera. hc is one
// value per frame, so the metric is 1-Lipschitz in xz and independent of the relief: the morph invariant at level
// borders depends only on the horizontal node diagonal (minR0)
double lodDistance(const Vec3& lo, const Vec3& hi, double hc)
{
    const double dx = std::max(0.0, std::max(lo.x(), -hi.x()));
    const double dz = std::max(0.0, std::max(lo.z(), -hi.z()));
    return std::sqrt(dx * dx + dz * dz + hc * hc);
}
}  // namespace

DescriptorSetLayout TerrainPass::setLayout()
{
    DescriptorSetLayout layout;
    const auto vfc =
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute;
    layout.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vfc));
    for (uint32_t b = 1; b <= 5; ++b)
        layout.addBinding(b, vk::DescriptorSetLayoutBinding(b, vk::DescriptorType::eStorageBuffer, 1, vfc));
    layout.addBinding(
        6, vk::DescriptorSetLayoutBinding(6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute));
    return layout;
}

void TerrainPass::init(VulkanBase& b, const RenderingFormats& formats, SceneLightingSet& lighting)
{
    destroy();
    base         = &b;
    lightingSet  = &lighting;
    sceneFormats = formats;
    samples      = int(formats.samples);
    frameLayout  = setLayout();
    frameLayout.create(base);
    skyLayout.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                                           vk::ShaderStageFlagBits::eFragment));
    skyLayout.create(base);
    createPipelines();

    probePipeline.init(*base, 1);
    probePipeline.addDescriptorSetLayout(setLayout(), 0);
    probePipeline.addPushConstantRange({vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)});
    probePipeline.shaderPipeline.loadCompute(base->device, "vulkan/scene/terrain_probe.comp");
    probePipeline.create();

    // The four grids in one index buffer
    std::vector<uint16_t> all;
    auto append = [&](Range& r, std::vector<uint16_t> v)
    {
        r.first = uint32_t(all.size());
        r.count = uint32_t(v.size());
        all.insert(all.end(), v.begin(), v.end());
    };
    append(fullSkirt, gridIndices(gridN, true));
    append(fullPlain, gridIndices(gridN, false));
    append(quarterSkirt, gridIndices(gridN / 2, true));
    append(quarterPlain, gridIndices(gridN / 2, false));
    indices.createBuffer(*base, all.size() * sizeof(uint16_t),
                         vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                         vk::MemoryPropertyFlagBits::eDeviceLocal);
    indices.stagedUpload(*base, all.size() * sizeof(uint16_t), all.data());

    for (Frame& f : frames)
    {
        f.uniforms.createBuffer(*base, sizeof(TerrainFrameUniforms), vk::BufferUsageFlagBits::eUniformBuffer,
                                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        f.set       = frameLayout.createRawDescriptorSet();
        f.shadowSet = frameLayout.createRawDescriptorSet();
        // S36a2: the shadow views' nodes (up to kMaxCascades views of maxNodes)
        f.shadowNodes.createBuffer(
            *base, size_t(kMaxCascades) * TerrainAtlas::maxNodes * sizeof(GpuNode),
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        // S36b: the atlas's nodes
        f.atlasSet = frameLayout.createRawDescriptorSet();
        f.atlasNodes.createBuffer(*base, size_t(kMaxAtlasNodes) * sizeof(GpuNode),
                                  vk::BufferUsageFlagBits::eStorageBuffer,
                                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
    for (size_t i = 0; i < frames.size(); ++i)
    {
        skySets[i] = skyLayout.createRawDescriptorSet();
        vk::DescriptorBufferInfo ubo(handle(frames[i].uniforms), 0, sizeof(TerrainFrameUniforms));
        base->device.updateDescriptorSets(
            vk::WriteDescriptorSet(skySets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &ubo), nullptr);
    }
    probeSet    = probePipeline.createRawDescriptorSet();
    initialized = true;
}

void TerrainPass::createNoShadowPipelines()
{
    // As createPipelines' scene and equal pipelines, with LIGHTING_NO_SHADOWS
    const std::string defines = "#define LIGHTING_NO_SHADOWS\n";
    PipelineInfo info;
    info.depthStencilState.depthCompareOp = vk::CompareOp::eGreaterOrEqual;  // reversed-Z
    const auto vf                         = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    auto create                           = [&](Pipeline& p, const PipelineInfo& pi)
    {
        p.init(*base, 2);
        p.addDescriptorSetLayout(setLayout(), 0);
        p.addDescriptorSetLayout(SceneLightingSet::layout(), 1);
        p.addPushConstantRange({vf, 0, sizeof(DrawPush)});
        p.shaderPipeline.loadGLSL(base->device,
                                  {{"vulkan/scene/terrain.vert", vk::ShaderStageFlagBits::eVertex, ""},
                                   {"vulkan/scene/terrain.frag", vk::ShaderStageFlagBits::eFragment, defines}});
        p.create(sceneFormats, pi);
    };
    create(sceneNoShadows, info);
    PipelineInfo equal                       = info;
    equal.depthStencilState.depthCompareOp   = vk::CompareOp::eEqual;
    equal.depthStencilState.depthWriteEnable = VK_FALSE;
    create(equalNoShadows, equal);
    noShadowPipelines = true;
}

void TerrainPass::createPipelines()
{
    // Set 0: the terrain's frame data; set 1: the lighting set (S36: shadeSurface of lighting.glsl)
    const std::string defines;
    PipelineInfo info;
    info.depthStencilState.depthCompareOp = vk::CompareOp::eGreaterOrEqual;  // reversed-Z
    const auto vf                         = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    auto create = [&](Pipeline& p, const std::vector<std::string>& shaders, const RenderingFormats& f, PipelineInfo pi,
                      const DescriptorSetLayout& set0, bool push)
    {
        p.init(*base, 2);
        p.addDescriptorSetLayout(set0, 0);
        p.addDescriptorSetLayout(SceneLightingSet::layout(), 1);
        if (push) p.addPushConstantRange({vf, 0, sizeof(DrawPush)});
        std::vector<std::tuple<std::string, vk::ShaderStageFlagBits, std::string>> modules = {
            {shaders[0], vk::ShaderStageFlagBits::eVertex, defines}};
        if (shaders.size() > 1) modules.emplace_back(shaders[1], vk::ShaderStageFlagBits::eFragment, defines);
        p.shaderPipeline.loadGLSL(base->device, modules);
        p.create(f, pi);
        creations++;
    };
    const std::vector<std::string> shaders = {"vulkan/scene/terrain.vert", "vulkan/scene/terrain.frag"};
    create(scenePipeline, shaders, sceneFormats, info, setLayout(), true);
    RenderingFormats depthOnly = sceneFormats;
    depthOnly.color.clear();
    create(depthPipeline, {"vulkan/scene/terrain.vert"}, depthOnly, info, setLayout(), true);
    PipelineInfo equal                       = info;
    equal.depthStencilState.depthCompareOp   = vk::CompareOp::eEqual;
    equal.depthStencilState.depthWriteEnable = VK_FALSE;
    create(equalPipeline, shaders, sceneFormats, equal, setLayout(), true);

    // The sky: binding 0 only (the frame's uniforms, also without a terrain set)
    DescriptorSetLayout sky0;
    sky0.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                                      vk::ShaderStageFlagBits::eFragment));
    PipelineInfo sky                       = info;
    sky.rasterizationState.cullMode        = vk::CullModeFlagBits::eNone;
    sky.depthStencilState.depthCompareOp   = vk::CompareOp::eEqual;
    sky.depthStencilState.depthWriteEnable = VK_FALSE;
    create(skyPipeline, {"vulkan/scene/sky.vert", "vulkan/scene/sky.frag"}, sceneFormats, sky, sky0, false);

    // S36a2: the terrain casters of the cascades (D32, LESS_OR_EQUAL, the slope bias dynamic, set 0 only)
    PipelineInfo shadow;
    shadow.depthStencilState.depthCompareOp   = vk::CompareOp::eLessOrEqual;
    shadow.depthStencilState.depthWriteEnable = VK_TRUE;
    shadow.rasterizationState.depthBiasEnable = VK_TRUE;
    shadow.dynamicStateEnables.push_back(vk::DynamicState::eDepthBias);
    shadowPipeline.init(*base, 1);
    shadowPipeline.addDescriptorSetLayout(setLayout(), 0);
    shadowPipeline.addPushConstantRange({vk::ShaderStageFlagBits::eVertex, 0, sizeof(ShadowPush)});
    shadowPipeline.shaderPipeline.loadGLSL(
        base->device, {{"vulkan/scene/lighting/terrain_shadow.vert", vk::ShaderStageFlagBits::eVertex, defines}});
    RenderingFormats shadowFormats;
    shadowFormats.depth = vk::Format::eD32Sfloat;
    shadowPipeline.create(shadowFormats, shadow);
}

void TerrainPass::destroy()
{
    if (!initialized) return;
    scenePipeline.destroy();
    depthPipeline.destroy();
    equalPipeline.destroy();
    skyPipeline.destroy();
    shadowPipeline.destroy();
    atlasPipeline.destroy();
    atlasPipelineCreated = false;
    sceneNoShadows.destroy();
    equalNoShadows.destroy();
    noShadowPipelines = false;
    probePipeline.destroy();
    indices.destroy();
    auto* pool = &base->descriptorPool;
    for (Frame& f : frames)
    {
        f.uniforms.destroy();
        f.shadowNodes.destroy();
        f.atlasNodes.destroy();
        for (vk::DescriptorSet* s : {&f.set, &f.shadowSet, &f.atlasSet})
        {
            auto set = *s;
            base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
            *s = nullptr;
        }
    }
    for (auto& sky : skySets)
    {
        auto set = sky;
        base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
        sky = nullptr;
    }
    auto set = probeSet;
    base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
    probeSet = nullptr;
    // The layouts after the sets (deferred, in order)
    DescriptorSetLayout fl = frameLayout, sl = skyLayout;
    base->deferDelete(
        [fl, sl]() mutable
        {
            fl.destroy();
            sl.destroy();
        });
    frameLayout = DescriptorSetLayout();
    skyLayout   = DescriptorSetLayout();
    initialized = false;
    frameValid  = false;
    lastAtlas   = nullptr;
}

void TerrainPass::skipFrame()
{
    frameValid = false;
    skyValid   = false;
    statistics = Stats();
    nodeInfos.clear();
    shadowStatistics.clear();
    for (Frame& f : frames) f.shadowRanges.clear();
}

bool TerrainPass::transfer(vk::CommandBuffer, uint32_t frameIndex, TerrainAtlas& atlas, const SceneCamera& camera,
                           const PrototypeSettings& settings, vec2 grassPhases, int width, int height)
{
    skipFrame();
    if (!camera.valid) return false;
    // A non-finite camera (no conversion to integers below): nothing this frame
    if (!camera.position.allFinite()) return false;
    // The terrain only for the set's space when it is the camera's; nothing before the whole pyramid is on the GPU
    const bool drawTerrain = atlas.hasSet() && atlas.pyramidResident() && atlas.space() == camera.space;
    const bool hasSea      = drawTerrain && atlas.seaLevel().has_value();
    lastFrame              = frameIndex;
    const Vec3& c          = camera.position;
    // The frame anchor: xz rounded down to 64 m (clamped to +-2^30 m before the conversion), refHeight to 64 m
    // (65536 units), clamped to the limit
    auto anchorOf = [](double v)
    { return int32_t(std::floor(std::min(std::max(v, -1073741824.0), 1073741824.0) / 64.0)) * 64; };
    const int32_t anchorX = anchorOf(c.x());
    const int32_t anchorZ = anchorOf(c.z());
    // The height clamped in double before the conversion (any finite y)
    const double refUnits =
        std::min(std::max(std::floor(c.y() / kHeightUnit / 65536.0) * 65536.0, -double(kTerrainHeightLimit)),
                 double(kTerrainHeightLimit));
    const int64_t ref = int64_t(refUnits);
    const Vec3 rel(c.x() - anchorX, c.y() - double(ref) * kHeightUnit, c.z() - anchorZ);
    // The morph start in [0.05, 0.95] (NaN: 0.05), for the range minimum and the shader alike: >= 1 would invert the
    // morph, a smaller one void minR0's guarantee
    float morphStart = settings.morphStart;
    if (!(morphStart >= 0.05f)) morphStart = 0.05f;
    morphStart = std::min(morphStart, 0.95f);

    if (drawTerrain)
    {
        lastAtlas              = &atlas;
        lastSetRebuilds        = atlas.stats().rebuilds;
        const TerrainLayout& L = atlas.layout();
        // The morph minimum. The S19a bound (2.5 leaf diagonals, more for a late morph start) and the CDLOD invariant
        // of the horizontal metric: at a level-k/k+1 border the fine node's box is within range_k, so the shared edge
        // is within range_k + its diagonal (sqrt(2) * leaf * 2^k) and must stay before the coarse level's morph start
        // range_k * (1 + morphStart): range0 >= sqrt(2) * leaf / morphStart, with a 50 % margin
        const double leaf  = 32.0 * L.baseSpacing;
        const double ms    = double(morphStart);
        const double minR0 = std::max(2.5 * leaf * 1.42 * std::max(1.0, 0.3 / std::max(1.0 - ms, 0.01)),
                                      1.5 * std::sqrt(2.0) * leaf / ms);
        range0             = std::max(double(settings.terrainLodRange), minR0);
        // The LOD metric's height: the camera height above the surface under it (one value per frame)
        const std::optional<double> ground = atlas.surfaceHeightEstimate(c.x(), c.z());
        lodHeight                          = ground ? std::abs(c.y() - *ground) : 0.0;

        Selection main;
        const std::array<Vec4, 6> planes = {camera.planes[0], camera.planes[1], camera.planes[2],
                                            camera.planes[3], camera.planes[4], Vec4(0, 0, 0, 1e30)};
        select(atlas, camera.position, planes, anchorX, anchorZ, main);
        fullNodes    = std::move(main.full);
        quarterNodes = std::move(main.quarter);
        fullInfos    = std::move(main.fullInfos);
        quarterInfos = std::move(main.quarterInfos);
        statistics   = main.stats;

        // The water plane: 400 km centred on the camera's xz rounded down to 4096 m, at the sea level, culled with its
        // own box
        waterNodes.clear();
        if (hasSea)
        {
            // The centre clamped like the anchor (+-2^30 m) before the conversion: the origins stay in int32_t
            auto centreOf = [](double v)
            { return int64_t(std::floor(std::min(std::max(v, -1073741824.0), 1073741824.0) / 4096.0)) * 4096; };
            const int64_t cx = centreOf(c.x()), cz = centreOf(c.z());
            const int32_t ox = int32_t(cx - 200000), oz = int32_t(cz - 200000);
            const double y = double(*atlas.seaLevel()) * kHeightUnit - c.y();
            const Vec3 lo(double(ox) - c.x(), y, double(oz) - c.z());
            const Vec3 hi(double(ox) + 400000 - c.x(), y, double(oz) + 400000 - c.z());
            if (!boxCulled(lo, hi, camera.planes))
            {
                GpuNode n;
                n.originX    = ox - anchorX;
                n.originZ    = oz - anchorZ;
                n.sampleX    = 400000 / (gridN / 2);
                n.levelFlags = flagWater;
                waterNodes.push_back(n);
                waterInfo         = TerrainNodeInfo();
                waterInfo.originX = ox;
                waterInfo.originZ = oz;
                waterInfo.water   = true;
                statistics.water  = true;
            }
        }

        // Instances: full, quarter, water; clipped to maxNodes (the water is kept)
        const size_t room = size_t(TerrainAtlas::maxNodes) - waterNodes.size();
        if (fullNodes.size() + quarterNodes.size() > room)
        {
            statistics.clipped = int(fullNodes.size() + quarterNodes.size() - room);
            std::cout << "TerrainPass: " << statistics.clipped << " nodes beyond " << TerrainAtlas::maxNodes
                      << " were clipped" << std::endl;
            if (fullNodes.size() > room)
            {
                fullNodes.resize(room);
                fullInfos.resize(room);
            }
            quarterNodes.resize(room - fullNodes.size());
            quarterInfos.resize(room - fullNodes.size());
        }
        numFull    = uint32_t(fullNodes.size());
        numQuarter = uint32_t(quarterNodes.size());
        numWater   = uint32_t(waterNodes.size());
        {
            auto* dst = static_cast<GpuNode*>(atlas.nodeBuffer(frameIndex).getMappedPointer());
            // An empty group's data() may be null: no memcpy (PR review)
            auto copy = [](GpuNode* to, const std::vector<GpuNode>& from)
            {
                if (!from.empty()) std::memcpy(to, from.data(), from.size() * sizeof(GpuNode));
            };
            copy(dst, fullNodes);
            copy(dst + numFull, quarterNodes);
            copy(dst + numFull + numQuarter, waterNodes);
        }
        nodeInfos = fullInfos;
        nodeInfos.insert(nodeInfos.end(), quarterInfos.begin(), quarterInfos.end());
        if (numWater) nodeInfos.push_back(waterInfo);
        statistics.nodes    = int(numFull + numQuarter);
        statistics.quarters = int(numQuarter);
    }

    // TerrainFrame (every frame with a camera: the sky reads it also without a terrain set)
    TerrainFrameUniforms& u = uniforms;
    u.viewProjRel           = camera.viewProj;
    u.invViewProjRel        = inverse(camera.viewProj);
    u.cameraRel             = vec4(float(rel.x()), float(rel.y()), float(rel.z()), 0);
    u.anchor                = ivec4(anchorX, anchorZ, int32_t(ref), hasSea ? *atlas.seaLevel() : 0);
    u.grid                  = ivec4(0, 0, 0, 0);
    u.flags                 = ivec4(hasSea ? 1 : 0, 0, settings.debugMode, 0);
    for (int k = 0; k < 8; ++k) u.levels[size_t(k)] = ivec4(0, 0, 0, 0);
    if (drawTerrain)
    {
        const TerrainLayout& L = atlas.layout();
        u.grid  = ivec4(L.samplesPerSide(), L.streamedLevels(), L.pyramidSamplesPerSide(), L.baseSpacing);
        u.flags = ivec4(hasSea ? 1 : 0, L.rootLevel(), settings.debugMode, 0);
        for (int k = 0; k < L.streamedLevels(); ++k)
            u.levels[size_t(k)] = ivec4(int(atlas.pageOffset(k)), L.tilesPerSide(k), 0, 0);
    }
    // S36: the light, the fog and the sky's colours are the lighting block's
    u.shading          = vec4(0, 0, settings.flatShading ? 1 : 0, camera.linear ? camera.preExposure : 0.0f);
    u.lod              = vec4(float(range0), morphStart, float(lodHeight), 0);
    u.grass            = vec4(settings.grassRadius, settings.grassCellSize, settings.grassDensity, float(samples));
    u.grassPhases      = vec4(grassPhases(0), grassPhases(1), settings.wind, settings.alphaCutoff);
    const float aspect = float(width) / float(std::max(height, 1));
    u.viewport         = vec4(aspect, 1, 1.0f / aspect, 1);
    std::memcpy(frames[frameIndex].uniforms.getMappedPointer(), &u, sizeof(u));
    skyValid = true;
    if (!drawTerrain) return false;
    writeDescriptors(frames[frameIndex], atlas, frameIndex);
    frameValid = true;
    return true;
}

void TerrainPass::select(const TerrainAtlas& atlas, const Vec3& camera, const std::array<Vec4, 6>& planes,
                         int32_t anchorX, int32_t anchorZ, Selection& out, const ShadowView* dedup)
{
    out                    = Selection();
    const TerrainLayout& L = atlas.layout();
    const int R            = L.rootLevel();
    std::vector<double> ranges(size_t(R) + 1);
    for (int k = 0; k <= R; ++k) ranges[size_t(k)] = range0 * std::ldexp(1.0, k);
    ranges[size_t(R)] = std::numeric_limits<double>::infinity();
    const int roots   = L.sizeChunks / 16;
    for (int z = 0; z < roots; ++z)
        for (int x = 0; x < roots; ++x)
            selectNode(atlas, camera, planes, R, x, z, anchorX, anchorZ, ranges, out, dedup);
}

bool TerrainPass::selectNode(const TerrainAtlas& atlas, const Vec3& c, const std::array<Vec4, 6>& planes, int level,
                             int32_t nx, int32_t nz, int32_t anchorX, int32_t anchorZ,
                             const std::vector<double>& ranges, Selection& out, const ShadowView* dedup)
{
    const TerrainLayout& L      = atlas.layout();
    const TerrainAtlas::Bound b = atlas.bound(level, nx, nz);
    // Not drawable at this level (no tile)
    if (!b.valid()) return true;
    const TerrainAtlas::Bound own = atlas.ownBound(level, nx, nz);
    const int size                = (32 * L.baseSpacing) << level;
    const double spacing          = double(L.baseSpacing << level);
    const double skirt            = std::max(double(own.hi - own.lo) * kHeightUnit, 2 * spacing);
    const int32_t ox = nx * size, oz = nz * size;
    // The box for the culling test reaches down by the skirt; the LOD distance is horizontal (lodDistance)
    const Vec3 lo(double(ox) - c.x(), double(b.lo) * kHeightUnit - skirt - c.y(), double(oz) - c.z());
    const Vec3 hi(double(ox + size) - c.x(), double(b.hi) * kHeightUnit - c.y(), double(oz + size) - c.z());
    const double d = lodDistance(lo, hi, lodHeight);
    if (d > ranges[size_t(level)]) return false;
    if (boxCulled(lo, hi, planes)) return true;
    // In the previous cascade's de-duplication volume: that cascade draws this node (the same main-view LOD) and its
    // receivers sample it (as the instances' rule, A5)
    if (dedup && dedup->dedupContains(lo, hi, true))
    {
        out.stats.deduplicated++;
        return true;
    }
    if (level == 0 || d > ranges[size_t(level - 1)])
    {
        addNode(atlas, level, ox, oz, false, anchorX, anchorZ, skirt, out);
        return true;
    }
    // Refined only if the children's tile (one tile holds all four) is GPU-resident
    const int child = level - 1;
    if (child < L.streamedLevels())
    {
        const int64_t tileSize = int64_t(256) << child;
        if (!atlas.tileResident(child, int32_t(int64_t(ox) / tileSize), int32_t(int64_t(oz) / tileSize)))
        {
            out.stats.refusedResidency++;
            addNode(atlas, level, ox, oz, false, anchorX, anchorZ, skirt, out);
            return true;
        }
    }
    const int half = size / 2;
    for (int q = 0; q < 4; ++q)
    {
        const int32_t cx = 2 * nx + (q & 1), cz = 2 * nz + (q >> 1);
        if (!selectNode(atlas, c, planes, child, cx, cz, anchorX, anchorZ, ranges, out, dedup))
        {
            // The child is out of its range: its area at this level's spacing, culled with this level's own heights
            // (what the quarter draws; the child's bound misses them when the levels' data differ, PR review)
            const int32_t qx = cx * half, qz = cz * half;
            const Vec3 clo(double(qx) - c.x(), double(own.lo) * kHeightUnit - skirt - c.y(), double(qz) - c.z());
            const Vec3 chi(double(qx + half) - c.x(), double(own.hi) * kHeightUnit - c.y(), double(qz + half) - c.z());
            if (!atlas.bound(child, cx, cz).valid() || boxCulled(clo, chi, planes)) continue;
            if (dedup && dedup->dedupContains(clo, chi, true))
            {
                out.stats.deduplicated++;
                continue;
            }
            addNode(atlas, level, qx, qz, true, anchorX, anchorZ, skirt, out);
        }
    }
    return true;
}

void TerrainPass::addNode(const TerrainAtlas& atlas, int level, int32_t ox, int32_t oz, bool quarter, int32_t anchorX,
                          int32_t anchorZ, double skirt, Selection& out)
{
    const int32_t spacing = atlas.layout().baseSpacing << level;
    GpuNode n;
    n.originX    = ox - anchorX;
    n.originZ    = oz - anchorZ;
    n.sampleX    = ox / spacing;
    n.sampleZ    = oz / spacing;
    n.levelFlags = uint32_t(level) | (quarter ? flagQuarter : 0u);
    n.skirtDepth = float(skirt);
    TerrainNodeInfo info;
    info.level   = level;
    info.originX = ox;
    info.originZ = oz;
    info.sampleX = n.sampleX;
    info.sampleZ = n.sampleZ;
    info.quarter = quarter;
    out.stats.nodesPerLevel[size_t(level)]++;
    if (quarter)
    {
        out.stats.quartersPerLevel[size_t(level)]++;
        out.stats.triangles += (gridN / 2) * (gridN / 2) * 2;
        out.quarter.push_back(n);
        out.quarterInfos.push_back(info);
    }
    else
    {
        out.stats.triangles += gridN * gridN * 2;
        out.full.push_back(n);
        out.fullInfos.push_back(info);
    }
}

void TerrainPass::selectForShadow(uint32_t frameIndex, const TerrainAtlas& atlas, const SceneCamera& camera,
                                  const std::vector<ShadowView>& views)
{
    Frame& f = frames[frameIndex];
    f.shadowRanges.clear();
    shadowStatistics.clear();
    if (!frameValid || views.empty()) return;
    const int32_t anchorX = uniforms.anchor(0), anchorZ = uniforms.anchor(1);
    auto* dst          = static_cast<GpuNode*>(f.shadowNodes.getMappedPointer());
    uint32_t used      = 0;
    const uint32_t cap = uint32_t(kMaxCascades) * TerrainAtlas::maxNodes;
    Selection sel;
    for (const ShadowView& v : views)
    {
        select(atlas, camera.position, v.planes(true), anchorX, anchorZ, sel, v.dedup ? &v : nullptr);
        Frame::ShadowRange r;
        r.first = used;
        r.full  = uint32_t(std::min<size_t>(sel.full.size(), cap - used));
        used += r.full;
        r.quarter = uint32_t(std::min<size_t>(sel.quarter.size(), cap - used));
        if (r.full) std::memcpy(dst + r.first, sel.full.data(), r.full * sizeof(GpuNode));
        if (r.quarter) std::memcpy(dst + r.first + r.full, sel.quarter.data(), r.quarter * sizeof(GpuNode));
        used += r.quarter;
        f.shadowRanges.push_back(r);
        ShadowStats st;
        st.nodes     = int(r.full + r.quarter);
        st.triangles    = int(r.full) * gridN * gridN * 2 + int(r.quarter) * (gridN / 2) * (gridN / 2) * 2;
        st.deduplicated = sel.stats.deduplicated;
        shadowStatistics.push_back(st);
    }
    // The set with the shadow nodes at binding 5 (the frame's set for the rest)
    std::array<vk::DescriptorBufferInfo, 6> infos = {
        vk::DescriptorBufferInfo(handle(f.uniforms), 0, sizeof(TerrainFrameUniforms)),
        vk::DescriptorBufferInfo(handle(lastAtlas->atlasBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(lastAtlas->pyramidBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(lastAtlas->pageBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(lastAtlas->maskBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(f.shadowNodes), 0, VK_WHOLE_SIZE)};
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(f.shadowSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &infos[0]);
    for (uint32_t b = 1; b <= 5; ++b)
        writes.emplace_back(f.shadowSet, b, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[b]);
    base->device.updateDescriptorSets(writes, nullptr);
}

void TerrainPass::renderShadow(vk::CommandBuffer cmd, uint32_t frameIndex, size_t view, const mat4& viewProj,
                               float slopeBias, const PrototypeSettings& s)
{
    Frame& f = frames[frameIndex];
    if (!frameValid || !s.terrain || view >= f.shadowRanges.size()) return;
    const Frame::ShadowRange r = f.shadowRanges[view];
    if (r.full + r.quarter == 0 || !shadowPipeline.bind(cmd)) return;
    shadowPipeline.bindRawDescriptorSet(cmd, f.shadowSet);
    cmd.bindIndexBuffer(handle(indices), 0, vk::IndexType::eUint16);
    cmd.setDepthBias(0.0f, 0.0f, slopeBias);
    auto draw = [&](const Range& range, int n, uint32_t count, uint32_t first)
    {
        ShadowPush push;
        push.viewProj = viewProj;
        push.gridN    = n;
        push.pad[0] = push.pad[1] = push.pad[2] = 0;
        shadowPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eVertex, sizeof(push), &push);
        cmd.drawIndexed(range.count, count, range.first, 0, first);
    };
    if (r.full > 0) draw(s.terrainSkirts ? fullSkirt : fullPlain, gridN, r.full, r.first);
    if (r.quarter > 0) draw(s.terrainSkirts ? quarterSkirt : quarterPlain, gridN / 2, r.quarter, r.first + r.full);
}

void TerrainPass::selectSphere(const TerrainAtlas& atlas, const Vec3& c, const Vec3& centre, double radius, int level,
                               int32_t nx, int32_t nz, int32_t anchorX, int32_t anchorZ, Selection& out, uint64_t& key)
{
    const TerrainLayout& L      = atlas.layout();
    const TerrainAtlas::Bound b = atlas.bound(level, nx, nz);
    if (!b.valid()) return;
    const TerrainAtlas::Bound own = atlas.ownBound(level, nx, nz);
    const int size                = (32 * L.baseSpacing) << level;
    const double spacing          = double(L.baseSpacing << level);
    const double skirt            = std::max(double(own.hi - own.lo) * kHeightUnit, 2 * spacing);
    const int32_t ox = nx * size, oz = nz * size;
    const Vec3 lo(double(ox) - c.x(), double(b.lo) * kHeightUnit - skirt - c.y(), double(oz) - c.z());
    const Vec3 hi(double(ox + size) - c.x(), double(b.hi) * kHeightUnit - c.y(), double(oz + size) - c.z());
    // The sphere against the node's box
    const Vec3 d = (lo - centre).cwiseMax(Vec3::Zero()).cwiseMax(centre - hi);
    if (d.squaredNorm() > radius * radius) return;
    // Refined while the children's tile is GPU-resident (the pyramid's levels always are)
    const int child = level - 1;
    bool refine     = level > 0;
    if (refine && child < L.streamedLevels())
    {
        const int64_t tileSize = int64_t(256) << child;
        refine = atlas.tileResident(child, int32_t(int64_t(ox) / tileSize), int32_t(int64_t(oz) / tileSize));
    }
    if (refine)
    {
        for (int q = 0; q < 4; ++q)
            selectSphere(atlas, c, centre, radius, child, 2 * nx + (q & 1), 2 * nz + (q >> 1), anchorX, anchorZ, out,
                         key);
        return;
    }
    addNode(atlas, level, ox, oz, false, anchorX, anchorZ, skirt, out);
    out.full.back().levelFlags |= flagFixed;
    // FNV-1a of the node
    for (int64_t v : {int64_t(level), int64_t(nx), int64_t(nz)})
    {
        key ^= uint64_t(v);
        key *= 1099511628211ull;
    }
}

void TerrainPass::selectForAtlas(uint32_t frameIndex, const TerrainAtlas& atlas, const SceneCamera& camera,
                                 const std::vector<std::pair<Vec3, double>>& spheres, std::vector<uint64_t>& keys)
{
    Frame& f = frames[frameIndex];
    f.atlasRanges.clear();
    keys.assign(spheres.size(), 0);
    atlasNodes = 0;
    if (!frameValid || spheres.empty()) return;
    const int32_t anchorX = uniforms.anchor(0), anchorZ = uniforms.anchor(1);
    auto* dst              = static_cast<GpuNode*>(f.atlasNodes.getMappedPointer());
    uint32_t used          = 0;
    const TerrainLayout& L = atlas.layout();
    const int R            = L.rootLevel();
    const int roots        = L.sizeChunks / 16;
    const int rootSize     = (32 * L.baseSpacing) << R;
    for (size_t i = 0; i < spheres.size(); ++i)
    {
        Selection sel;
        uint64_t key       = 1469598103934665603ull;
        const Vec3& centre = spheres[i].first;
        const double r     = spheres[i].second;
        // Only the roots the sphere touches
        const Vec3 w = camera.position + centre;
        const int x0 = std::max(0, int(std::floor((w.x() - r) / rootSize))),
                  x1 = std::min(roots - 1, int(std::floor((w.x() + r) / rootSize)));
        const int z0 = std::max(0, int(std::floor((w.z() - r) / rootSize))),
                  z1 = std::min(roots - 1, int(std::floor((w.z() + r) / rootSize)));
        for (int z = z0; z <= z1; ++z)
            for (int x = x0; x <= x1; ++x)
                selectSphere(atlas, camera.position, centre, r, R, x, z, anchorX, anchorZ, sel, key);
        Frame::ShadowRange range;
        range.first = used;
        range.full  = uint32_t(std::min<size_t>(sel.full.size(), kMaxAtlasNodes - used));
        if (range.full) std::memcpy(dst + range.first, sel.full.data(), range.full * sizeof(GpuNode));
        used += range.full;
        f.atlasRanges.push_back(range);
        keys[i] = sel.full.empty() ? 0 : key;
    }
    atlasNodes                                    = int(used);
    std::array<vk::DescriptorBufferInfo, 6> infos = {
        vk::DescriptorBufferInfo(handle(f.uniforms), 0, sizeof(TerrainFrameUniforms)),
        vk::DescriptorBufferInfo(handle(lastAtlas->atlasBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(lastAtlas->pyramidBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(lastAtlas->pageBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(lastAtlas->maskBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(f.atlasNodes), 0, VK_WHOLE_SIZE)};
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(f.atlasSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &infos[0]);
    for (uint32_t b = 1; b <= 5; ++b)
        writes.emplace_back(f.atlasSet, b, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[b]);
    base->device.updateDescriptorSets(writes, nullptr);
}

void TerrainPass::createAtlasPipeline()
{
    // S36b: the terrain casters of the shadow atlas (D16, the linear depth from the fragment shader, no bias)
    PipelineInfo atlasInfo;
    atlasInfo.depthStencilState.depthCompareOp   = vk::CompareOp::eLessOrEqual;
    atlasInfo.depthStencilState.depthWriteEnable = VK_TRUE;
    atlasPipeline.init(*base, 1);
    atlasPipeline.addDescriptorSetLayout(setLayout(), 0);
    atlasPipeline.addPushConstantRange(
        {vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(AtlasPush)});
    const std::string atlasDefines = "#define ATLAS_DEPTH\n";
    atlasPipeline.shaderPipeline.loadGLSL(
        base->device, {{"vulkan/scene/lighting/terrain_shadow.vert", vk::ShaderStageFlagBits::eVertex, atlasDefines},
                       {"vulkan/scene/lighting/atlas_depth.frag", vk::ShaderStageFlagBits::eFragment, atlasDefines}});
    RenderingFormats atlasFormats;
    atlasFormats.depth = vk::Format::eD16Unorm;
    atlasPipeline.create(atlasFormats, atlasInfo);
    atlasPipelineCreated = true;
}

void TerrainPass::renderAtlas(vk::CommandBuffer cmd, uint32_t frameIndex, size_t sphere, const mat4& viewProj,
                              const vec4& light, const PrototypeSettings& s)
{
    Frame& f = frames[frameIndex];
    if (!frameValid || !s.terrain || sphere >= f.atlasRanges.size()) return;
    const Frame::ShadowRange r = f.atlasRanges[sphere];
    if (r.full == 0) return;
    // On first use (a scene without shadowed lights compiles none)
    if (!atlasPipelineCreated) createAtlasPipeline();
    if (!atlasPipeline.bind(cmd)) return;
    atlasPipeline.bindRawDescriptorSet(cmd, f.atlasSet);
    cmd.bindIndexBuffer(handle(indices), 0, vk::IndexType::eUint16);
    AtlasPush push;
    push.viewProj = viewProj;
    push.gridN    = gridN;
    push.pad[0] = push.pad[1] = push.pad[2] = 0;
    push.light                              = light;
    atlasPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, sizeof(push),
                               &push);
    const Range& range = s.terrainSkirts ? fullSkirt : fullPlain;
    cmd.drawIndexed(range.count, r.full, range.first, 0, r.first);
}

void TerrainPass::writeDescriptors(Frame& f, TerrainAtlas& atlas, uint32_t frameIndex)
{
    std::array<vk::DescriptorBufferInfo, 6> infos = {
        vk::DescriptorBufferInfo(handle(f.uniforms), 0, sizeof(TerrainFrameUniforms)),
        vk::DescriptorBufferInfo(handle(atlas.atlasBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(atlas.pyramidBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(atlas.pageBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(atlas.maskBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(atlas.nodeBuffer(frameIndex)), 0, VK_WHOLE_SIZE)};
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(f.set, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &infos[0]);
    for (uint32_t b = 1; b <= 5; ++b)
        writes.emplace_back(f.set, b, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[b]);
    base->device.updateDescriptorSets(writes, nullptr);
}

void TerrainPass::render(vk::CommandBuffer cmd, uint32_t frameIndex, Mode mode, const PrototypeSettings& s,
                         bool shadows)
{
    if (!frameValid) return;
    const bool terrain = s.terrain && numFull + numQuarter > 0;
    const bool water   = s.water && numWater > 0;
    if (!terrain && !water) return;
    if (!shadows && mode != Mode::Depth && !noShadowPipelines) createNoShadowPipelines();
    Pipeline& p = mode == Mode::Depth   ? depthPipeline
                  : mode == Mode::Equal ? (shadows ? equalPipeline : equalNoShadows)
                                        : (shadows ? scenePipeline : sceneNoShadows);
    if (!p.bind(cmd)) return;
    p.bindRawDescriptorSet(cmd, frames[frameIndex].set);
    p.bindRawDescriptorSet(cmd, lightingSet->set(frameIndex), 1);
    cmd.bindIndexBuffer(handle(indices), 0, vk::IndexType::eUint16);
    const auto stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    auto draw         = [&](const Range& r, int n, uint32_t count, uint32_t first)
    {
        DrawPush push{n, 0};
        p.pushConstant(cmd, stages, sizeof(push), &push);
        cmd.drawIndexed(r.count, count, r.first, 0, first);
    };
    if (terrain)
    {
        if (numFull > 0) draw(s.terrainSkirts ? fullSkirt : fullPlain, gridN, numFull, 0);
        if (numQuarter > 0) draw(s.terrainSkirts ? quarterSkirt : quarterPlain, gridN / 2, numQuarter, numFull);
    }
    if (water) draw(quarterPlain, gridN / 2, numWater, numFull + numQuarter);
}

void TerrainPass::renderSky(vk::CommandBuffer cmd, uint32_t frameIndex, const PrototypeSettings& s)
{
    if (!skyValid || !s.sky || !skyPipeline.bind(cmd)) return;
    skyPipeline.bindRawDescriptorSet(cmd, skySets[frameIndex]);
    skyPipeline.bindRawDescriptorSet(cmd, lightingSet->set(frameIndex), 1);
    cmd.draw(3, 1, 0, 0);
}

std::vector<vec4> TerrainPass::probeVertices(const std::vector<TerrainProbe>& probes)
{
    std::vector<vec4> out;
    SAIGA_ASSERT(lastAtlas && frameValid, "probeVertices: no terrain frame");
    // The frame's nodes and descriptors refer to its set: a cleared or new set needs a new frame (PR review)
    SAIGA_ASSERT(lastAtlas->hasSet() && lastAtlas->stats().rebuilds == lastSetRebuilds,
                 "probeVertices: the terrain set changed after the last terrain frame");
    if (probes.empty()) return out;
    base->device.waitIdle();
    Buffer buffer;
    const size_t bytes = probes.size() * sizeof(ProbeGpu);
    buffer.createBuffer(*base, bytes, vk::BufferUsageFlagBits::eStorageBuffer,
                        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto* gpu = static_cast<ProbeGpu*>(buffer.getMappedPointer());
    for (size_t i = 0; i < probes.size(); ++i)
    {
        SAIGA_ASSERT(probes[i].node >= 0 && size_t(probes[i].node) < nodeInfos.size(), "probeVertices: no such node");
        gpu[i] = ProbeGpu{{uint32_t(probes[i].node), uint32_t(probes[i].gridX), uint32_t(probes[i].gridZ),
                           probes[i].normal ? 1u : 0u},
                          {0, 0, 0, 0}};
    }
    // The last frame's set with the probes
    writeDescriptors(frames[lastFrame], *lastAtlas, lastFrame);
    {
        std::array<vk::DescriptorBufferInfo, 7> infos = {
            vk::DescriptorBufferInfo(handle(frames[lastFrame].uniforms), 0, sizeof(TerrainFrameUniforms)),
            vk::DescriptorBufferInfo(handle(lastAtlas->atlasBuffer()), 0, VK_WHOLE_SIZE),
            vk::DescriptorBufferInfo(handle(lastAtlas->pyramidBuffer()), 0, VK_WHOLE_SIZE),
            vk::DescriptorBufferInfo(handle(lastAtlas->pageBuffer()), 0, VK_WHOLE_SIZE),
            vk::DescriptorBufferInfo(handle(lastAtlas->maskBuffer()), 0, VK_WHOLE_SIZE),
            vk::DescriptorBufferInfo(handle(lastAtlas->nodeBuffer(lastFrame)), 0, VK_WHOLE_SIZE),
            vk::DescriptorBufferInfo(handle(buffer), 0, VK_WHOLE_SIZE)};
        std::vector<vk::WriteDescriptorSet> writes;
        writes.emplace_back(probeSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &infos[0]);
        for (uint32_t b = 1; b <= 6; ++b)
            writes.emplace_back(probeSet, b, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[b]);
        base->device.updateDescriptorSets(writes, nullptr);
    }
    auto& q               = base->mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    if (probePipeline.bind(cmd))
    {
        probePipeline.bindRawDescriptorSet(cmd, probeSet);
        const uint32_t count = uint32_t(probes.size());
        probePipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(count), &count);
        cmd.dispatch((count + 63) / 64, 1, 1);
        vk::MemoryBarrier2 b(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
                             vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, b, nullptr, nullptr));
    }
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    for (size_t i = 0; i < probes.size(); ++i)
        out.push_back(vec4(gpu[i].position[0], gpu[i].position[1], gpu[i].position[2], gpu[i].position[3]));
    buffer.destroyImmediately();
    return out;
}

}  // namespace Vulkan
}  // namespace Saiga
