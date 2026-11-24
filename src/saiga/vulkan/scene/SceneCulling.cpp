/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/SceneCulling.h"

#include "saiga/vulkan/Renderer.h"
#include "saiga/vulkan/scene/InstanceBuffer.h"
#include "saiga/vulkan/scene/MeshPool.h"
#include "saiga/vulkan/scene/SceneLightingSet.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <tuple>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
static_assert(VulkanRenderer::framesInFlight == 2, "SceneCulling: one set of frame resources per frame in flight");

constexpr uint32_t workChunkSize = 256;
constexpr uint32_t dynamicChunk  = 0xFFFFFFFFu;
constexpr uint32_t commandStride = sizeof(vk::DrawIndexedIndirectCommand);

// The bindings of the scene's descriptor set (shader/vulkan/scene/instance_common.glsl)
enum Binding : uint32_t
{
    bChunks = 0,
    bInstances,
    bDynamics,
    bMeshes,
    bOffsets,
    bCounts,
    bScratch,
    bCommands,
    bBucketLods,
    bStatistics,
    bCompacted,
    bMaterials,
    bPalette,
    bindingCount
};

struct CountPush
{
    float planes[5][4];
    float alpha;
    uint32_t meshCount;
    uint32_t bucketsPerView;
    uint32_t view;
    uint32_t chunkCount;
};
struct ScanPush
{
    uint32_t bucketCount;
    uint32_t meshCount;
};
struct WritePush
{
    uint32_t chunkCount;
};
struct DrawPush
{
    mat4 viewProj;
    float alpha;
    float preExposure;
    uint32_t linear;
    uint32_t pad;
};
static_assert(sizeof(CountPush) == 100 && sizeof(DrawPush) == 80, "push constants of the scene shaders");

// shader/vulkan/scene/bucket_scan.comp: Statistics. The tracked mesh IDs are written by the host (the frame's list),
// the counts by the scan
struct GpuStatistics
{
    uint32_t instancesPerLod[4];
    uint32_t nonEmptyBuckets;
    uint32_t trackedCount;
    uint32_t pad[2];
    uint32_t trackedMeshes[maxTrackedMeshes];
    uint32_t trackedPerLod[maxTrackedMeshes][4];
};
static_assert(sizeof(GpuStatistics) == 32 + maxTrackedMeshes * 20, "Statistics of bucket_scan.comp");

DescriptorSetLayout sceneLayout()
{
    DescriptorSetLayout layout;
    const auto stages =
        vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    for (uint32_t b = 0; b < bindingCount; ++b)
        layout.addBinding(b, vk::DescriptorSetLayoutBinding(b, vk::DescriptorType::eStorageBuffer, 1, stages));
    return layout;
}

void memoryBarrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
{
    vk::MemoryBarrier2 b(srcStage, srcAccess, dstStage, dstAccess);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, b, nullptr, nullptr));
}

vk::Buffer handle(Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}

// The AABB [lo, hi] (camera-relative) is outside a plane or farther than maxDistance
bool aabbCulled(const Vec3& lo, const Vec3& hi, const std::array<Vec4, 5>& planes, double maxDistance)
{
    Vec3 nearest = Vec3::Zero().cwiseMax(lo).cwiseMin(hi);
    if (nearest.norm() > maxDistance) return true;
    for (const Vec4& p : planes)
    {
        // The corner farthest along the normal
        Vec3 c(p(0) >= 0 ? hi(0) : lo(0), p(1) >= 0 ? hi(1) : lo(1), p(2) >= 0 ? hi(2) : lo(2));
        if (p(0) * c(0) + p(1) * c(1) + p(2) * c(2) + p(3) < 0) return true;
    }
    return false;
}
}  // namespace

void SceneCulling::init(VulkanBase& b, const RenderingFormats& formats, const RenderSceneParameters& params,
                        SceneLightingSet& lighting)
{
    base         = &b;
    lightingSet  = &lighting;
    sceneFormats = formats;
    SAIGA_ASSERT(base->enabledFeatures.drawIndirectFirstInstance,
                 "RenderScene: the drawIndirectFirstInstance feature is required");
    multiDraw         = params.allowMultiDrawIndirect && base->enabledFeatures.multiDrawIndirect;
    dynamicCapacity   = params.dynamicCapacity;
    const auto limits = base->physicalDevice.getProperties().limits;
    // The scene layout's storage buffers are visible to every scene stage (Vulkan's minimum per stage is only 4), and
    // the lighting set's (S36) to the draws' stages
    const uint32_t storagePerStage = bindingCount + SceneLightingSet::kStorageBuffers;
    if (limits.maxPerStageDescriptorStorageBuffers < storagePerStage ||
        limits.maxDescriptorSetStorageBuffers < storagePerStage)
        SAIGA_EXIT_ERROR("RenderScene: " + std::to_string(storagePerStage) +
                         " storage buffers per stage are required (maxPerStageDescriptorStorageBuffers: " +
                         std::to_string(limits.maxPerStageDescriptorStorageBuffers) + ")");
    maxDrawIndirectCount = std::max(1u, limits.maxDrawIndirectCount);
    maxWorkGroups        = limits.maxComputeWorkGroupCount[0];
    maxWorkGroupsY       = limits.maxComputeWorkGroupCount[1];
    if (params.workGroupLimit > 0) maxWorkGroups = std::min(maxWorkGroups, params.workGroupLimit);
    storageBytes = limits.maxStorageBufferRange;
    if (params.storageBufferLimit > 0) storageBytes = std::min<uint64_t>(storageBytes, params.storageBufferLimit);

    auto initCompute = [&](ComputePipeline& p, const char* shader, uint32_t pushSize)
    {
        p.init(*base, 1);
        p.addDescriptorSetLayout(sceneLayout());
        p.addPushConstantRange({vk::ShaderStageFlagBits::eCompute, 0, pushSize});
        p.shaderPipeline.loadCompute(base->device, shader);
        p.create();
    };
    initCompute(countPipeline, "vulkan/scene/instance_count.comp", sizeof(CountPush));
    initCompute(scanPipeline, "vulkan/scene/bucket_scan.comp", sizeof(ScanPush));
    initCompute(writePipeline, "vulkan/scene/instance_write.comp", sizeof(WritePush));

    createDrawPipelines();

    // The sets are compatible with every pipeline's layout (identically defined set layouts)
    for (Frame& f : frames) f.set = countPipeline.createRawDescriptorSet();
}

void SceneCulling::destroyDrawPipelines()
{
    for (auto& p : drawPipelines) p.destroy();
    for (auto& p : prepassPipelines) p.destroy();
    for (auto& p : equalPipelines) p.destroy();
}

void SceneCulling::createDrawPipelines()
{
    const RenderingFormats& formats = sceneFormats;
    // The draw pipelines: set 0 the scene's buffers, set 1 the lighting set (SceneLightingSet). shaders: vertex and
    // (optional) fragment shader, injected with the defines
    auto initDraw =
        [&](Pipeline& p, const std::vector<std::string>& shaders, const std::string& defines, PipelineInfo info)
    {
        info.addVertexBinding(0, sizeof(MeshVertex), vk::VertexInputRate::eVertex,
                              {{0, 0, vk::Format::eR32G32B32Sfloat, uint32_t(offsetof(MeshVertex, position))},
                               {1, 0, vk::Format::eR32Uint, uint32_t(offsetof(MeshVertex, normal))},
                               {2, 0, vk::Format::eR32Uint, uint32_t(offsetof(MeshVertex, color))},
                               {3, 0, vk::Format::eR32G32Sfloat, uint32_t(offsetof(MeshVertex, uv))}});
        p.init(*base, 2);
        p.addDescriptorSetLayout(sceneLayout(), 0);
        p.addDescriptorSetLayout(SceneLightingSet::layout(), 1);
        p.addPushConstantRange(
            {vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPush)});
        std::vector<std::tuple<std::string, vk::ShaderStageFlagBits, std::string>> modules = {
            {shaders[0], vk::ShaderStageFlagBits::eVertex, defines}};
        if (shaders.size() > 1) modules.emplace_back(shaders[1], vk::ShaderStageFlagBits::eFragment, defines);
        p.shaderPipeline.loadGLSL(base->device, modules);
        creations++;
        return info;
    };
    RenderingFormats depthOnly = formats;
    depthOnly.color.clear();
    const std::vector<std::string> flatShaders = {"vulkan/scene/instance.vert", "vulkan/scene/flat.frag"};
    // One class's pipeline: opaque, alpha-masked, two-sided, alpha-masked two-sided, prototype foliage
    // (MeshPool::pipelineClassIndex)
    auto createClass = [&](Pipeline& p, int c, const std::string& defines, const std::string& foliageDefines)
    {
        const bool foliage  = c == int(PipelineClass::PrototypeFoliage) - 1;
        const bool twoSided = c >= 2;
        PipelineInfo info;
        info.depthStencilState.depthCompareOp = vk::CompareOp::eGreaterOrEqual;  // reversed-Z
        info.rasterizationState.cullMode      = twoSided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack;
        // The leaf cards: alpha to coverage (with MSAA); the opaque parts write alpha 1
        if (foliage) info.multisampleState.alphaToCoverageEnable = VK_TRUE;
        const std::vector<std::string> shaders =
            foliage
                ? std::vector<std::string>{"vulkan/scene/prototype_foliage.vert", "vulkan/scene/prototype_foliage.frag"}
                : flatShaders;
        info = initDraw(p, shaders, foliage ? foliageDefines : defines, info);
        p.create(formats, info);
    };
    // The opaque classes after the pre-pass: EQUAL, no depth write
    auto createEqual = [&](Pipeline& p, int k, const std::string& defines)
    {
        PipelineInfo info;
        info.depthStencilState.depthCompareOp   = vk::CompareOp::eEqual;
        info.depthStencilState.depthWriteEnable = VK_FALSE;
        info.rasterizationState.cullMode        = k == 1 ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack;
        info                                    = initDraw(p, flatShaders, defines, info);
        p.create(formats, info);
    };
    for (int c = 0; c < pipelineClassCount; ++c) createClass(drawPipelines[c], c, "", "");
    for (int k = 0; k < 2; ++k) createEqual(equalPipelines[k], k, "");
    // The opaque classes in the pre-pass: depth only, the vertex shader alone (invariant gl_Position: the same depth
    // as every variant's)
    for (int k = 0; k < 2; ++k)
    {
        PipelineInfo info;
        info.depthStencilState.depthCompareOp = vk::CompareOp::eGreaterOrEqual;
        info.rasterizationState.cullMode      = k == 1 ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack;
        info                                  = initDraw(prepassPipelines[k], {"vulkan/scene/instance.vert"}, "", info);
        prepassPipelines[k].create(depthOnly, info);
    }
}

void SceneCulling::destroy()
{
    if (!base) return;
    countPipeline.destroy();
    scanPipeline.destroy();
    writePipeline.destroy();
    destroyDrawPipelines();
    for (Frame& f : frames)
    {
        f.offsets.destroy();
        f.chunks.destroy();
        f.dynamics.destroy();
        f.stats.destroy();
        f.staging.destroy();
        if (f.set)
        {
            auto* pool = &base->descriptorPool;
            auto set   = f.set;
            base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
            f.set = nullptr;
        }
    }
    counts.destroy();
    scratch.destroy();
    compacted.destroy();
    commands.destroy();
    bucketLods.destroy();
    base = nullptr;
}

size_t SceneCulling::storageSize(const Buffer& b, size_t bytes, const char* what) const
{
    if (bytes > storageBytes)
    {
        SAIGA_EXIT_ERROR("RenderScene: the " + std::string(what) + " need " + std::to_string(bytes) +
                         " bytes, more than maxStorageBufferRange (" + std::to_string(storageBytes) + " bytes)");
    }
    return size_t(std::min<uint64_t>(std::max<size_t>(bytes, size_t(b.size()) * 2), storageBytes));
}

void SceneCulling::ensureHost(Buffer& b, size_t bytes, vk::BufferUsageFlags usage, const char* what)
{
    bytes = std::max<size_t>(bytes, 256);
    if (b && b.size() >= bytes) return;
    // The frame slot's previous frame has completed; max(2x, required) within the limit
    b.createBuffer(*base, storageSize(b, bytes, what), usage,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
}

void SceneCulling::ensureDevice(Buffer& b, size_t bytes, vk::BufferUsageFlags usage, bool& recreated, const char* what)
{
    bytes = std::max<size_t>(bytes, 256);
    if (b && b.size() >= bytes) return;
    // No copy: the content is rewritten every frame (or the static fields by the caller); the old buffer is released
    // after the frames in flight
    b.createBuffer(*base, storageSize(b, bytes, what), usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
    recreated = true;
}

void SceneCulling::recordTransfer(vk::CommandBuffer cmd, uint32_t frameIndex, MeshPool& meshes,
                                  InstanceBuffer& instances, const SceneCamera& camera)
{
    Frame& f = frames[frameIndex];


    // 1. Growth copies, made visible to the uploads that follow (an updated range of a moved set)
    bool copied = meshes.recordGrowth(cmd);
    copied      = instances.recordGrowth(cmd) || copied;
    if (copied)
        memoryBarrier(cmd, Stage::eAllTransfer, Access::eTransferWrite, Stage::eAllTransfer,
                      Access::eTransferWrite | Access::eTransferRead);

    // 2. The work chunks: visible sets (bounds against the frustum and the meshes' end distances), then the dynamics
    chunkList.clear();
    if (camera.valid)
    {
        for (const auto& [id, s] : instances.sets())
        {
            if (s.region.space != camera.space || s.used == 0 || !s.hasBounds) continue;
            const Vec3 rel = regionOrigin(s.region) - camera.position;
            if (aabbCulled(rel + s.boundsMin, rel + s.boundsMax, camera.planes, s.maxEndDistance)) continue;
            const uint32_t first = instances.setOffset(id);
            for (uint32_t i = 0; i < s.used; i += workChunkSize)
                chunkList.push_back({first + i, std::min(workChunkSize, s.used - i), s.slot, 0});
        }
        const uint32_t dynamicCount = uint32_t(instances.dynamics().size());
        for (uint32_t i = 0; i < dynamicCount; i += workChunkSize)
            chunkList.push_back({i, std::min(workChunkSize, dynamicCount - i), dynamicChunk, 0});
    }
    if (meshes.buckets().empty()) chunkList.clear();
    // A 2D dispatch beyond maxComputeWorkGroupCount[0] (the scratch buffer's limit is reached long before y's)
    if ((uint64_t(chunkList.size()) + maxWorkGroups - 1) / maxWorkGroups > maxWorkGroupsY)
        SAIGA_EXIT_ERROR("RenderScene: " + std::to_string(chunkList.size()) +
                         " work chunks exceed maxComputeWorkGroupCount");
    f.chunkCount = uint32_t(chunkList.size());
    f.culled     = f.chunkCount > 0;

    // 3. The frame's host visible buffers
    const auto storage = vk::BufferUsageFlagBits::eStorageBuffer;
    ensureHost(f.chunks, chunkList.size() * sizeof(WorkChunk), storage, "work chunks");
    if (!chunkList.empty()) std::memcpy(f.chunks.getMappedPointer(), chunkList.data(), chunkList.size() * 16);

    const auto& slots = instances.slots();
    ensureHost(f.offsets, slots.size() * 32, storage, "offset table entries");
    {
        auto* o = static_cast<float*>(f.offsets.getMappedPointer());
        for (size_t i = 0; i < slots.size(); ++i)
        {
            float* e = o + 8 * i;
            std::fill(e, e + 8, 0.0f);
            const auto& s = slots[i];
            Vec3 origin   = Vec3::Zero();
            bool valid    = false;
            if (camera.valid && s.kind == InstanceBuffer::Slot::Kind::Region && s.region.space == camera.space)
            {
                origin = regionOrigin(s.region);
                valid  = true;
            }
            else if (camera.valid && s.kind == InstanceBuffer::Slot::Kind::Chunk && s.chunk.space == camera.space)
            {
                origin = chunkOrigin(s.chunk);
                valid  = true;
            }
            if (!valid) continue;  // w = 0: rejected by the count pass
            // origin - camera in double, split into float hi + lo
            const Vec3 d = origin - camera.position;
            for (int k = 0; k < 3; ++k)
            {
                const float hi = float(d(k));
                e[k]           = hi;
                e[4 + k]       = float(d(k) - double(hi));
            }
            e[3] = 1;
        }
    }

    const auto& dyn = instances.dynamics();
    ensureHost(f.dynamics, std::max<size_t>(dyn.size(), dynamicCapacity) * sizeof(InstanceBuffer::GpuDynamic), storage,
               "dynamic instances");
    if (!dyn.empty()) instances.writeDynamics(static_cast<InstanceBuffer::GpuDynamic*>(f.dynamics.getMappedPointer()));

    ensureHost(f.stats, sizeof(GpuStatistics), storage, "statistics");
    {
        // The frame's tracked meshes (read by the scan and by readStatistics), zero counts (the scan overwrites them)
        GpuStatistics g;
        std::memset(&g, 0, sizeof(g));
        g.trackedCount = uint32_t(trackedMeshes.size());
        for (size_t i = 0; i < trackedMeshes.size(); ++i) g.trackedMeshes[i] = trackedMeshes[i];
        std::memcpy(f.stats.getMappedPointer(), &g, sizeof(g));
    }

    // 4. The shared device buffers (recreated when too small)
    const size_t buckets = meshes.buckets().size();
    bool layoutBuffers = false, other = false;
    const auto devStorage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    ensureDevice(counts, buckets * 4, devStorage | vk::BufferUsageFlagBits::eTransferSrc, other, "bucket counts");
    ensureDevice(commands, buckets * commandStride, devStorage | vk::BufferUsageFlagBits::eIndirectBuffer,
                 layoutBuffers, "indirect commands");
    ensureDevice(bucketLods, buckets * 4, devStorage, layoutBuffers, "bucket LODs");
    ensureDevice(scratch, size_t(f.chunkCount) * workChunkSize * 8, devStorage, other, "scratch entries");
    ensureDevice(compacted, (size_t(instances.capacity()) + std::max<size_t>(dyn.size(), dynamicCapacity)) * 8,
                 devStorage, other, "compacted entries");

    // 5. Uploads: meshes, instances, tables, the commands' static fields and bucket LODs after a layout change
    meshes.queueUploads(batch);
    instances.queueUploads(batch);
    if (layoutBuffers || uploadedLayout != meshes.layoutVersion())
    {
        bucketCount = uint32_t(buckets);
        if (buckets > 0)
        {
            std::vector<vk::DrawIndexedIndirectCommand> cmds(buckets);
            std::vector<uint32_t> lods(buckets);
            for (uint32_t b = 0; b < buckets; ++b)
            {
                cmds[b] = meshes.command(b);
                lods[b] = meshes.buckets()[b].lod;
            }
            batch.add(commands, 0, cmds.data(), buckets * commandStride);
            batch.add(bucketLods, 0, lods.data(), buckets * 4);
        }
        uploadedLayout = meshes.layoutVersion();
    }
    batch.record(*base, cmd, f.staging);

    writeDescriptors(f, meshes, instances);

    // The draws' snapshot: a content change in the render callback (a layout change, a growth of the vertex or index
    // buffer) takes effect in the next frame, whose transfer uploads it
    for (int c = 0; c < pipelineClassCount; ++c)
        f.ranges[c] = {meshes.classRanges()[c].first, meshes.classRanges()[c].count};
    f.vertices = handle(meshes.vertexBuffer());
    f.indices  = handle(meshes.indexBuffer());
    f.commands = handle(commands);
}

void SceneCulling::writeDescriptors(Frame& f, MeshPool& meshes, InstanceBuffer& instances)
{
    // The frame slot's set is not used by a frame in flight
    std::array<vk::DescriptorBufferInfo, bindingCount> infos;
    infos[bChunks]     = vk::DescriptorBufferInfo(handle(f.chunks), 0, VK_WHOLE_SIZE);
    infos[bInstances]  = vk::DescriptorBufferInfo(handle(instances.instanceBuffer()), 0, VK_WHOLE_SIZE);
    infos[bDynamics]   = vk::DescriptorBufferInfo(handle(f.dynamics), 0, VK_WHOLE_SIZE);
    infos[bMeshes]     = vk::DescriptorBufferInfo(handle(meshes.meshRecordBuffer()), 0, VK_WHOLE_SIZE);
    infos[bOffsets]    = vk::DescriptorBufferInfo(handle(f.offsets), 0, VK_WHOLE_SIZE);
    infos[bCounts]     = vk::DescriptorBufferInfo(handle(counts), 0, VK_WHOLE_SIZE);
    infos[bScratch]    = vk::DescriptorBufferInfo(handle(scratch), 0, VK_WHOLE_SIZE);
    infos[bCommands]   = vk::DescriptorBufferInfo(handle(commands), 0, VK_WHOLE_SIZE);
    infos[bBucketLods] = vk::DescriptorBufferInfo(handle(bucketLods), 0, VK_WHOLE_SIZE);
    infos[bStatistics] = vk::DescriptorBufferInfo(handle(f.stats), 0, sizeof(GpuStatistics));
    infos[bCompacted]  = vk::DescriptorBufferInfo(handle(compacted), 0, VK_WHOLE_SIZE);
    infos[bMaterials]  = vk::DescriptorBufferInfo(handle(meshes.materialBuffer()), 0, VK_WHOLE_SIZE);
    // The fragment shader clamps the palette index to the range's length
    infos[bPalette] = vk::DescriptorBufferInfo(handle(meshes.paletteBuffer()), 0, meshes.paletteCount() * 8);
    std::array<vk::WriteDescriptorSet, bindingCount> writes;
    for (uint32_t b = 0; b < bindingCount; ++b)
        writes[b] = vk::WriteDescriptorSet(f.set, b, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[b]);
    base->device.updateDescriptorSets(writes, nullptr);
}

void SceneCulling::recordCull(vk::CommandBuffer cmd, uint32_t frameIndex, MeshPool& meshes, const SceneCamera& camera)
{
    Frame& f = frames[frameIndex];
    if (!f.culled) return;

    // The previous frame's reads (draw, vertex shader) and writes (passes) of the shared buffers come first
    memoryBarrier(cmd, Stage::eDrawIndirect | Stage::eVertexShader | Stage::eComputeShader | Stage::eAllTransfer,
                  Access::eShaderStorageWrite | Access::eTransferWrite, Stage::eAllTransfer | Stage::eComputeShader,
                  Access::eTransferWrite | Access::eShaderStorageRead | Access::eShaderStorageWrite);
    cmd.fillBuffer(handle(counts), 0, VK_WHOLE_SIZE, 0);
    memoryBarrier(cmd, Stage::eAllTransfer, Access::eTransferWrite, Stage::eComputeShader,
                  Access::eShaderStorageRead | Access::eShaderStorageWrite);

    // Count
    if (countPipeline.bind(cmd))
    {
        CountPush push;
        for (int k = 0; k < 5; ++k)
            for (int j = 0; j < 4; ++j) push.planes[k][j] = float(camera.planes[k](j));
        push.alpha          = camera.alpha;
        push.meshCount      = meshes.meshTableSize();
        push.bucketsPerView = bucketCount;
        push.view           = 0;
        push.chunkCount     = f.chunkCount;
        countPipeline.bindRawDescriptorSet(cmd, f.set);
        countPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        dispatchChunks(cmd, f.chunkCount);
    }
    memoryBarrier(cmd, Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eComputeShader,
                  Access::eShaderStorageRead | Access::eShaderStorageWrite);

    // Scan
    if (scanPipeline.bind(cmd))
    {
        ScanPush push{bucketCount, meshes.meshTableSize()};
        scanPipeline.bindRawDescriptorSet(cmd, f.set);
        scanPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        cmd.dispatch(1, 1, 1);
    }
    memoryBarrier(cmd, Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eComputeShader,
                  Access::eShaderStorageRead | Access::eShaderStorageWrite);

    // Write
    if (writePipeline.bind(cmd))
    {
        WritePush push{f.chunkCount};
        writePipeline.bindRawDescriptorSet(cmd, f.set);
        writePipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        dispatchChunks(cmd, f.chunkCount);
    }
    // The draws read the commands and the compacted entries; the host reads the statistics after the frame
    memoryBarrier(cmd, Stage::eComputeShader, Access::eShaderStorageWrite,
                  Stage::eDrawIndirect | Stage::eVertexShader | Stage::eHost,
                  Access::eIndirectCommandRead | Access::eShaderStorageRead | Access::eHostRead);
}

uint32_t SceneCulling::drawClass(vk::CommandBuffer cmd, uint32_t frameIndex, int klass, Pipeline& p,
                                 const SceneCamera& camera)
{
    Frame& f         = frames[frameIndex];
    const auto range = f.ranges[klass];
    if (range.count == 0) return 0;
    if (!p.bind(cmd)) return 0;
    DrawPush push;
    push.viewProj    = camera.viewProj;
    push.alpha       = camera.alpha;
    push.preExposure = camera.preExposure;
    push.linear      = camera.linear ? 1 : 0;
    push.pad         = 0;
    p.bindRawDescriptorSet(cmd, f.set, 0);
    p.bindRawDescriptorSet(cmd, lightingSet->set(frameIndex), 1);
    p.pushConstant(cmd, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, sizeof(push), &push);
    cmd.bindVertexBuffers(0, f.vertices, vk::DeviceSize(0));
    cmd.bindIndexBuffer(f.indices, 0, vk::IndexType::eUint32);
    // The snapshot of recordTransfer (the layout of the uploaded commands), not the mesh pool's current state
    const vk::Buffer commandBuffer = f.commands;
    uint32_t calls                 = 0;
    if (multiDraw)
    {
        for (uint32_t first = 0; first < range.count; first += maxDrawIndirectCount)
        {
            const uint32_t n = std::min(maxDrawIndirectCount, range.count - first);
            cmd.drawIndexedIndirect(commandBuffer, vk::DeviceSize(range.first + first) * commandStride, n,
                                    commandStride);
            calls++;
        }
    }
    else
    {
        for (uint32_t b = 0; b < range.count; ++b)
        {
            cmd.drawIndexedIndirect(commandBuffer, vk::DeviceSize(range.first + b) * commandStride, 1, commandStride);
            calls++;
        }
    }
    return calls;
}

uint32_t SceneCulling::recordPrepass(vk::CommandBuffer cmd, uint32_t frameIndex, const SceneCamera& camera)
{
    Frame& f = frames[frameIndex];
    if (!f.culled) return 0;
    // The opaque classes: opaque (index 0) and two-sided (index 2)
    return drawClass(cmd, frameIndex, 0, prepassPipelines[0], camera) +
           drawClass(cmd, frameIndex, 2, prepassPipelines[1], camera);
}

uint32_t SceneCulling::recordDraw(vk::CommandBuffer cmd, uint32_t frameIndex, const SceneCamera& camera, bool prepass)
{
    Frame& f = frames[frameIndex];
    if (!f.culled) return 0;
    uint32_t calls = 0;
    for (int c = 0; c < pipelineClassCount; ++c)
    {
        // The prototype foliage reads the foliage texture (binding 1, written only with it): not drawn without it (a
        // frame with such instances asserts in RenderScene with SAIGA_ASSERTS). Only the draw is skipped: the GPU
        // culling still processes those instances, and its statistics (instancesPerLod, the tracked meshes) count them
        if (c == int(PipelineClass::PrototypeFoliage) - 1 && !foliageTexture) continue;
        Pipeline* p = &drawPipelines[c];
        if (prepass && c == 0) p = &equalPipelines[0];
        if (prepass && c == 2) p = &equalPipelines[1];
        calls += drawClass(cmd, frameIndex, c, *p, camera);
    }
    return calls;
}

void SceneCulling::dispatchChunks(vk::CommandBuffer cmd, uint32_t chunkCount) const
{
    // Chunk c = y * x-count + x; the shaders skip c >= chunkCount
    if (chunkCount == 0) return;
    const uint32_t x = std::min(chunkCount, maxWorkGroups);
    const uint32_t y = (chunkCount + x - 1) / x;
    cmd.dispatch(x, y, 1);
}

void SceneCulling::setTrackedMeshes(const std::vector<MeshId>& meshes)
{
    if (meshes.size() > size_t(maxTrackedMeshes))
    {
        SAIGA_EXIT_ERROR("RenderScene::trackMeshStatistics: " + std::to_string(meshes.size()) +
                         " meshes, at most maxTrackedMeshes (" + std::to_string(maxTrackedMeshes) + ")");
        return;
    }
    trackedMeshes = meshes;
}

void SceneCulling::readStatistics(uint32_t frameIndex, RenderStatistics& out)
{
    Frame& f = frames[frameIndex];
    GpuStatistics g;
    std::memset(&g, 0, sizeof(g));
    if (f.stats) std::memcpy(&g, f.stats.getMappedPointer(), sizeof(g));
    for (int i = 0; i < 4; ++i) out.instancesPerLod[i] = g.instancesPerLod[i];
    out.nonEmptyBuckets = g.nonEmptyBuckets;
    out.trackedMeshes.clear();
    for (uint32_t t = 0; t < std::min<uint32_t>(g.trackedCount, maxTrackedMeshes); ++t)
    {
        RenderStatistics::TrackedMesh m;
        m.mesh = MeshId(g.trackedMeshes[t]);
        for (int l = 0; l < 4; ++l) m.instancesPerLod[l] = g.trackedPerLod[t][l];
        out.trackedMeshes.push_back(m);
    }
}

}  // namespace Vulkan
}  // namespace Saiga
