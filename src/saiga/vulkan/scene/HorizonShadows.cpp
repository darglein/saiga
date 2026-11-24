/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/HorizonShadows.h"

#include "saiga/vulkan/pipeline/DescriptorSetLayout.h"
#include "saiga/vulkan/scene/SceneLighting.h"
#include "saiga/vulkan/scene/SceneLightingSet.h"
#include "saiga/vulkan/scene/TerrainAtlas.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

// horizon_map.comp's march (its cell sizes 32 m << level, its 16 / 32 m steps) assumes the pyramid's 32 m spacing
// (nit 18): a change of the spacing must change the shader
static_assert(TerrainLayout::pyramidSpacing == 32, "horizon_map.comp assumes 32 m pyramid cells");

namespace
{
constexpr double kPi = 3.14159265358979323846;

// shader/vulkan/scene/lighting/horizon_maxmip.comp
struct ChainPush
{
    int32_t samples, level, srcOffset, srcSize, dstOffset, dstSize;
};
// shader/vulkan/scene/lighting/horizon_map.comp
struct MapPush
{
    int32_t corner[2];
    int32_t resolution, layer, row0, rows;
    float texel;
    int32_t refUnits;
    float towardsLight[2];
    float tanEl, casterDistance;
    int32_t cells, levels;
    float spaceSize, pad;
    int32_t offsets[12];
};
static_assert(sizeof(MapPush) == 112, "Push of horizon_map.comp");

constexpr int kMaxLevels = 12;

vk::Buffer handle(Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}

void memoryBarrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
{
    vk::MemoryBarrier2 b(srcStage, srcAccess, dstStage, dstAccess);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, b, nullptr, nullptr));
}

// The angle between two directions (degrees)
double angleDeg(const Vec3& a, const Vec3& b)
{
    const double c = std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0);
    return std::acos(c) * 180.0 / kPi;
}
}  // namespace

void HorizonShadows::init(VulkanBase& b)
{
    destroy();
    base = &b;
    // Linear filtering of RG32F is optional: else nearest with manual bilinear weights (shadows.glsl)
    const auto features = base->physicalDevice.getFormatProperties(vk::Format::eR32G32Sfloat).optimalTilingFeatures;
    linearFilter        = bool(features & vk::FormatFeatureFlagBits::eSampledImageFilterLinear);
    SamplerParameters sp;
    sp.setFilter(linearFilter ? vk::Filter::eLinear : vk::Filter::eNearest);
    sp.setAddressMode(vk::SamplerAddressMode::eClampToEdge);
    sampler.create(*base, sp);
}

void HorizonShadows::createPipelines()
{
    if (pipelinesCreated) return;
    pipelinesCreated = true;
    const auto c     = vk::ShaderStageFlagBits::eCompute;
    {
        DescriptorSetLayout l;
        l.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, c));
        l.addBinding(1, vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, c));
        chainPipeline.init(*base, 1);
        chainPipeline.addDescriptorSetLayout(l);
        chainPipeline.addPushConstantRange({c, 0, sizeof(ChainPush)});
        chainPipeline.shaderPipeline.loadCompute(base->device, "vulkan/scene/lighting/horizon_maxmip.comp");
        chainPipeline.create();
        for (auto& set : chainSets) set = chainPipeline.createRawDescriptorSet();
    }
    {
        DescriptorSetLayout l;
        l.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, c));
        l.addBinding(1, vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageImage, 1, c));
        mapPipeline.init(*base, 1);
        mapPipeline.addDescriptorSetLayout(l);
        mapPipeline.addPushConstantRange({c, 0, sizeof(MapPush)});
        mapPipeline.shaderPipeline.loadCompute(base->device, "vulkan/scene/lighting/horizon_map.comp");
        mapPipeline.create();
        for (auto& set : mapSets) set = mapPipeline.createRawDescriptorSet();
    }
}

void HorizonShadows::destroy()
{
    if (!base) return;
    chainPipeline.destroy();
    mapPipeline.destroy();
    pipelinesCreated = false;
    auto* pool = &base->descriptorPool;
    for (auto* sets : {&chainSets, &mapSets})
        for (vk::DescriptorSet& s : *sets)
        {
            auto set = s;
            if (set) base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
            s = nullptr;
        }
    chain.destroy();
    if (maps) maps->destroy();
    maps.reset();
    sampler.destroy();
    // The chain and the maps are gone: a later init builds them again (CodeRabbit #85 round 2: the cached chain
    // identity skipped the rebuild of the released buffer)
    invalidate();
    base = nullptr;
}

void HorizonShadows::invalidate()
{
    for (MapTransform& t : transforms) t = MapTransform();
    computing     = false;
    restartedMap  = false;
    rowsDone      = 0;
    blend         = 1;
    blendStart    = 1;
    chainRebuilds = ~uint64_t(0);
    chainVersion  = ~uint64_t(0);
    chainPyramid  = nullptr;
    chainSpace    = 0;
}

void HorizonShadows::ensureImages(int resolution)
{
    if (maps && int(maps->width) == resolution) return;
    if (maps) maps->destroy();
    maps = std::make_unique<Texture2DArray>();
    maps->create(
        *base, uint32_t(resolution), uint32_t(resolution), 3, vk::Format::eR32G32Sfloat,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc);
    // eGeneral for good: written by the compute pass, sampled by the scene (the transform says which layer is valid)
    auto& q               = base->mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    vk::ImageMemoryBarrier2 b(Stage::eNone, Access::eNone, Stage::eAllCommands,
                              Access::eShaderStorageWrite | Access::eShaderSampledRead, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eGeneral, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                              maps->image(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 3));
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, b));
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    maps->setImageLayout(vk::ImageLayout::eGeneral);
    for (MapTransform& t : transforms) t = MapTransform();
    computing    = false;
    restartedMap = false;
    blend        = 1;
}

vk::DescriptorImageInfo HorizonShadows::descriptor() const
{
    return vk::DescriptorImageInfo(vk::Sampler(sampler), maps->view(), vk::ImageLayout::eGeneral);
}

void HorizonShadows::buildChain(vk::CommandBuffer cmd, const TerrainAtlas& atlas, uint32_t frameIndex)
{
    createPipelines();
    vk::DescriptorSet chainSet = chainSets[frameIndex];
    const int samples          = atlas.layout().pyramidSamplesPerSide();
    levelOffsets.clear();
    levelSizes.clear();
    int size = samples - 1, offset = 0;
    while (int(levelSizes.size()) < kMaxLevels)
    {
        levelOffsets.push_back(offset);
        levelSizes.push_back(size);
        offset += size * size;
        if (size == 1) break;
        size = (size + 1) / 2;
    }
    const size_t bytes = size_t(offset) * 4;
    if (!chain || chain.size() < bytes)
        chain.createBuffer(*base, bytes, vk::BufferUsageFlagBits::eStorageBuffer,
                           vk::MemoryPropertyFlagBits::eDeviceLocal);
    TerrainAtlas& a                               = const_cast<TerrainAtlas&>(atlas);
    std::array<vk::DescriptorBufferInfo, 2> infos = {
        vk::DescriptorBufferInfo(handle(a.pyramidBuffer()), 0, VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo(handle(chain), 0, VK_WHOLE_SIZE)};
    std::array<vk::WriteDescriptorSet, 2> writes = {
        vk::WriteDescriptorSet(chainSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[0]),
        vk::WriteDescriptorSet(chainSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[1])};
    base->device.updateDescriptorSets(writes, nullptr);
    // The pyramid's uploads (transfer) and the previous frame's march (compute) before the writes
    memoryBarrier(cmd, Stage::eAllTransfer | Stage::eComputeShader, Access::eTransferWrite | Access::eShaderStorageRead,
                  Stage::eComputeShader, Access::eShaderStorageRead | Access::eShaderStorageWrite);
    if (!chainPipeline.bind(cmd)) return;
    chainPipeline.bindRawDescriptorSet(cmd, chainSet);
    for (size_t l = 0; l < levelSizes.size(); ++l)
    {
        ChainPush push{
            samples,         int32_t(l),   l > 0 ? levelOffsets[l - 1] : 0, l > 0 ? levelSizes[l - 1] : samples,
            levelOffsets[l], levelSizes[l]};
        chainPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        cmd.dispatch(uint32_t(levelSizes[l] + 15) / 16, uint32_t(levelSizes[l] + 15) / 16, 1);
        memoryBarrier(cmd, Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eComputeShader,
                      Access::eShaderStorageRead | Access::eShaderStorageWrite);
    }
    // The level of the march's steps at 10 km (horizon_map.comp: cells of at least twice the step t / 16)
    levelAt10km =
        std::min(int(std::ceil(std::log2(2.0 * (10000.0 / 16.0) / double(TerrainLayout::pyramidSpacing)) - 1e-4)),
                 int(levelSizes.size()) - 1);
}

HorizonShadows::MapTransform HorizonShadows::targetTransform(const Vec3& camera, const Vec3& light,
                                                             const LightingParameters& p) const
{
    MapTransform t;
    t.valid                = true;
    t.resolution           = std::clamp(p.horizonResolution, 16, 8192);
    t.extent               = std::max(double(p.horizonExtent), 64.0);
    t.texel                = t.extent / double(t.resolution);
    t.light                = light.normalized();
    t.casterDistance       = p.cascadeCasterDistance;
    const MapTransform& f  = transforms[size_t(front)];
    const MapTransform& bk = transforms[size_t(back)];
    // The anchor of the newest map (the one in progress, else the front) while the camera stays in its inner half
    const MapTransform& anchorOf = computing && bk.valid ? bk : f;
    if (anchorOf.valid && anchorOf.resolution == t.resolution && anchorOf.extent == t.extent)
    {
        const double cx = anchorOf.cornerX + 0.5 * t.extent, cz = anchorOf.cornerZ + 0.5 * t.extent;
        if (std::abs(camera.x() - cx) <= 0.25 * t.extent && std::abs(camera.z() - cz) <= 0.25 * t.extent)
        {
            t.cornerX  = anchorOf.cornerX;
            t.cornerZ  = anchorOf.cornerZ;
            t.refUnits = anchorOf.refUnits;
            return t;
        }
    }
    // A new anchor: the camera's xz minus half the extent, snapped to the texel grid in integer metres (clamped like
    // the terrain anchor), the camera height rounded down to 64 m
    auto corner = [&](double v)
    {
        const double c = std::floor((std::clamp(v, -1073741824.0, 1073741824.0) - 0.5 * t.extent) / t.texel) * t.texel;
        return int32_t(std::floor(c));
    };
    t.cornerX       = corner(camera.x());
    t.cornerZ       = corner(camera.z());
    const double hu = std::floor(std::clamp(camera.y(), -1e6, 1e6) / kHeightUnit / 65536.0) * 65536.0;
    t.refUnits      = int64_t(hu);
    return t;
}

void HorizonShadows::recordRows(vk::CommandBuffer cmd, const MapTransform& t, int layer, int row0, int rows,
                                const TerrainAtlas& atlas, uint32_t frameIndex)
{
    createPipelines();
    vk::DescriptorSet mapSet                      = mapSets[frameIndex];
    std::array<vk::DescriptorBufferInfo, 1> infos = {vk::DescriptorBufferInfo(handle(chain), 0, VK_WHOLE_SIZE)};
    vk::DescriptorImageInfo image(nullptr, maps->view(), vk::ImageLayout::eGeneral);
    std::array<vk::WriteDescriptorSet, 2> writes = {
        vk::WriteDescriptorSet(mapSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[0]),
        vk::WriteDescriptorSet(mapSet, 1, 0, 1, vk::DescriptorType::eStorageImage, &image)};
    base->device.updateDescriptorSets(writes, nullptr);
    if (!mapPipeline.bind(cmd)) return;
    mapPipeline.bindRawDescriptorSet(cmd, mapSet);
    MapPush push;
    std::memset(&push, 0, sizeof(push));
    push.corner[0]  = t.cornerX;
    push.corner[1]  = t.cornerZ;
    push.resolution = t.resolution;
    push.layer      = layer;
    push.row0       = row0;
    push.rows       = rows;
    push.texel      = float(t.texel);
    push.refUnits   = int32_t(
        std::clamp<int64_t>(t.refUnits, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()));
    const Vec2 h(t.light.x(), t.light.z());
    const double hn      = h.norm();
    push.towardsLight[0] = hn > 1e-9 ? float(h.x() / hn) : 1.0f;
    push.towardsLight[1] = hn > 1e-9 ? float(h.y() / hn) : 0.0f;
    // At the zenith no terrain shadows: an elevation so steep that nothing reaches
    push.tanEl          = hn > 1e-6 ? float(t.light.y() / hn) : 1e6f;
    push.casterDistance = t.casterDistance;
    push.cells          = levelSizes.empty() ? 0 : levelSizes[0];
    push.levels         = int32_t(levelSizes.size());
    push.spaceSize      = float(atlas.layout().sizeChunks) * 256.0f;
    for (size_t l = 0; l < levelOffsets.size() && l < size_t(kMaxLevels); ++l) push.offsets[l] = levelOffsets[l];
    mapPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
    cmd.dispatch(uint32_t(t.resolution + 63) / 64, uint32_t(rows), 1);
}

int HorizonShadows::update(vk::CommandBuffer cmd, const Input& in)
{
    lastRows                    = 0;
    const LightingParameters& p = in.parameters;
    if (!p.horizonShadows)
    {
        // Off: the maps are released (the memory), every map starts again
        if (maps)
        {
            maps->destroy();
            maps.reset();
        }
        for (MapTransform& t : transforms) t = MapTransform();
        computing    = false;
        restartedMap = false;
        return 0;
    }
    if (in.terrainCleared) invalidate();
    if (!in.atlas || !in.hasLight) return 0;
    const int res = std::clamp(p.horizonResolution, 16, 8192);
    ensureImages(res);
    // A new terrain set (another space, a cleared set) invalidates the maps; a changed pyramid rebuilds the chain and
    // recomputes (the old maps stay shown meanwhile)
    const TerrainAtlas& atlas = *in.atlas;
    const uint64_t rebuilds = atlas.stats().rebuilds, version = atlas.pyramidVersion();
    if (rebuilds != chainRebuilds || &atlas != chainPyramid)
    {
        for (MapTransform& t : transforms) t = MapTransform();
        computing = false;
        blend     = 1;
    }
    const bool full = in.force || requestedFull;
    if (computing && full)
    {
        // A full update replaces the map in progress, in this frame (CodeRabbit #85 round 2: without a due change it
        // was not started again), on the newest chain
        computing    = false;
        restartedMap = false;
        requested    = true;
    }
    // After the one restart of a map, a later edit waits for the next map (the map in progress keeps one terrain
    // version: a rebuilt chain would mix versions in its rows; CodeRabbit #85 round 2)
    const bool changed = rebuilds != chainRebuilds || version != chainVersion || &atlas != chainPyramid;
    if (changed && !(computing && restartedMap))
    {
        buildChain(cmd, atlas, in.frameIndex);
        chainRebuilds = rebuilds;
        chainVersion  = version;
        chainPyramid  = &atlas;
        chainSpace    = atlas.space();
        requested     = true;
        // A map in progress restarts on the new chain (its rows before the change would keep the old terrain, and the
        // edit would show only after a second complete map; S36a2 review, finding 14), once per map: with an edit in
        // every frame (terraforming) a map restarted on each would never complete; a later edit is in the next map
        if (computing && !restartedMap)
        {
            restarts++;
            restartedMap = true;
            computing    = false;
        }
    }
    // The light's angular speed (the rows per frame follow it); a jump of 5 degrees or more (a time change, a new
    // light) is no speed: it starts over from the base rows per frame (S36a2 review, finding 6d: the speed of the
    // frames before a jump decayed over about 60 frames)
    const double moved = angleDeg(lastLight, in.light);
    degreesPerFrame    = moved < 5 ? std::max(moved, degreesPerFrame * 0.9) : 0.0;
    lastLight          = in.light;

    const MapTransform target = targetTransform(in.camera, in.light, p);
    const MapTransform& f     = transforms[size_t(front)];
    const bool sameGrid       = f.valid && f.resolution == target.resolution && f.extent == target.extent;
    const bool due            = requested || !sameGrid || f.cornerX != target.cornerX || f.cornerZ != target.cornerZ ||
                     angleDeg(f.light, in.light) > 0.1 || f.casterDistance != target.casterDistance;
    if (!computing && due)
    {
        transforms[size_t(back)] = target;
        rowsDone                 = 0;
        blendStart               = blend;
        computing                = true;
        requested                = false;
        recomputes++;
    }
    // The maps' readers: the scene's shaders (the grass samples them per vertex) and the previews
    const auto readers = SceneLightingSet::kReaderStages;
    if (computing)
    {
        const int needed = int(std::ceil(double(res) * degreesPerFrame / 0.1));
        int rows         = full ? res : std::max(std::max(p.horizonRowsPerFrame, 1), needed);
        rows             = std::min(rows, res - rowsDone);
        // The previous frames' reads of the maps before this frame's writes into the back map
        memoryBarrier(cmd, readers | Stage::eAllTransfer, Access::eShaderSampledRead | Access::eTransferRead,
                      Stage::eComputeShader, Access::eShaderStorageWrite);
        recordRows(cmd, transforms[size_t(back)], back, rowsDone, rows, atlas, in.frameIndex);
        rowsDone += rows;
        lastRows = rows;
        blend    = blendStart + (1 - blendStart) * float(rowsDone) / float(res);
        if (rowsDone >= res)
        {
            // previous <- front <- back; the blend restarts (the shown map is unchanged), at once with a full update or
            // the first map
            const bool show = full || !transforms[size_t(front)].valid;
            const int old   = previous;
            previous        = front;
            front           = back;
            back            = old;
            computing       = false;
            restartedMap    = false;
            completed++;
            lastComplete  = in.frame;
            blend         = show ? 1.0f : 0.0f;
            requestedFull = false;
        }
        memoryBarrier(cmd, Stage::eComputeShader, Access::eShaderStorageWrite, readers | Stage::eAllTransfer,
                      Access::eShaderSampledRead | Access::eTransferRead);
    }
    else if (blend < 1)
        blend = std::min(1.0f, blend + float(std::max(p.horizonRowsPerFrame, 1)) / float(res));
    return lastRows;
}

void HorizonShadows::fillBlock(SceneLighting& block, const Vec3& camera, SpaceId space, const LightingParameters& p,
                               bool hasLight, bool terrainDrawn) const
{
    block.horizonPrev     = vec4(0, 0, 0, -1);
    block.horizonFront    = vec4(0, 0, 0, -1);
    block.horizonRef      = vec4(0, 0, 1, 0);
    block.horizonParams   = vec4(0, 0, 0, 0);
    const MapTransform& f = transforms[size_t(front)];
    if (!p.horizonShadows || !hasLight || !maps || !f.valid || space != chainSpace) return;
    auto map = [&](const MapTransform& t, int layer)
    {
        if (!t.valid || t.resolution != f.resolution) return vec4(0, 0, 0, -1);
        return vec4(float(double(t.cornerX) - camera.x()), float(double(t.cornerZ) - camera.z()), float(1.0 / t.texel),
                    float(layer));
    };
    block.horizonPrev   = map(transforms[size_t(previous)], previous);
    block.horizonFront  = map(f, front);
    auto ref            = [&](const MapTransform& t) { return float(double(t.refUnits) * kHeightUnit - camera.y()); };
    block.horizonRef    = vec4(ref(transforms[size_t(previous)]), ref(f), blend, float(f.resolution));
    const Vec3 l        = f.light;
    const double hn     = Vec2(l.x(), l.z()).norm();
    block.horizonParams = vec4(1, float(hn > 1e-6 ? l.y() / hn : 1e6),
                               p.terrainCastsInCascades && p.sunShadows && terrainDrawn ? 1 : 0, linearFilter ? 1 : 0);
}

HorizonStatistics HorizonShadows::statistics() const
{
    HorizonStatistics s;
    const MapTransform& f = transforms[size_t(front)];
    s.on                  = maps != nullptr;
    s.valid               = f.valid;
    s.computing           = computing;
    s.resolution          = f.resolution;
    s.extent              = f.extent;
    s.texel               = f.texel;
    s.corner              = ivec2(f.cornerX, f.cornerZ);
    s.reference           = double(f.refUnits) * kHeightUnit;
    s.light               = f.light.cast<float>();
    s.rowsDone            = computing ? rowsDone : 0;
    s.rowsThisFrame       = lastRows;
    s.blend               = blend;
    s.recomputes          = recomputes;
    s.restarts            = restarts;
    s.completed           = completed;
    s.lastComplete        = lastComplete;
    s.levelAt10km         = levelAt10km;
    s.memoryBytes         = memoryBytes();
    s.chainBytes          = chain ? uint64_t(chain.size()) : 0;
    return s;
}

uint64_t HorizonShadows::memoryBytes() const
{
    uint64_t b = chain ? uint64_t(chain.size()) : 0;
    if (maps) b += uint64_t(maps->width) * maps->height * 3 * 8;
    return b;
}

std::vector<vec2> HorizonShadows::readLayerForTests(int layer)
{
    std::vector<vec2> out;
    if (!maps) return out;
    base->device.waitIdle();
    const uint32_t w = maps->width, h = maps->height;
    const size_t bytes = size_t(w) * h * 8;
    Buffer host;
    host.createBuffer(*base, bytes, vk::BufferUsageFlagBits::eTransferDst,
                      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto& q               = base->mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    {
        Memory::SafeAccessor acc(*host.memoryLocation());
        vk::BufferImageCopy region(0, 0, 0,
                                   vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, uint32_t(layer), 1),
                                   {0, 0, 0}, {w, h, 1});
        cmd.copyImageToBuffer(maps->image(), vk::ImageLayout::eGeneral, host.memoryLocation()->data, region);
    }
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    out.resize(size_t(w) * h);
    std::memcpy(out.data(), host.getMappedPointer(), bytes);
    host.destroyImmediately();
    return out;
}

}  // namespace Vulkan
}  // namespace Saiga
