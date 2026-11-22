/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"
#include "saiga/vulkan/scene/SceneLighting.h"
#include "saiga/vulkan/scene/UploadBatch.h"
#include "saiga/vulkan/texture/Texture.h"

#include <array>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class MeshPool;
class InstanceBuffer;
class SceneLightingSet;
class ShadowPipelineTable;

// The camera of a frame (computed by RenderScene in double)
struct SceneCamera
{
    bool valid    = false;
    SpaceId space = 0;
    Vec3 position = Vec3::Zero();
    // Camera-relative frustum planes, inward normals: dot(n, p) + w >= -r is inside (left, right, bottom, top,
    // near)
    std::array<Vec4, 5> planes = {Vec4::Zero(), Vec4::Zero(), Vec4::Zero(), Vec4::Zero(), Vec4::Zero()};
    mat4 viewProj              = mat4::Identity();
    float alpha                = 1;
    float preExposure          = 1;
    bool linear                = false;
};

/**
 * The GPU culling and the instance draws of RenderScene (S18, docs/engine_design.md §10.2):
 *   1. CPU: the instance sets' bounds against the frustum and their meshes' end distances -> work chunks of at most 256
 *      instances {first, count, slot} over each visible set's used size; the dynamics chunked with slot 0xFFFFFFFF;
 *   2. count pass (instance_count.comp, one workgroup per chunk, a 2D dispatch beyond maxComputeWorkGroupCount[0]):
 * rejections, frustum, LOD, atomicAdd into the bucket's count; the bucket and the returned slot are stored in the
 * scratch entry;
 *   3. scan (bucket_scan.comp): instanceCount and firstInstance of every bucket's indirect command, the statistics
 *      (with the per-LOD counts of the tracked meshes, whose IDs the frame's statistics buffer carries);
 *   4. write pass (instance_write.comp): the compacted entries uvec2(instance index | dynamic bit, slot | LOD << 24);
 *   5. draw: one vkCmdDrawIndexedIndirect per pipeline class and view over its buckets (split at
 *      maxDrawIndirectCount), or one per bucket without multiDrawIndirect. S19a: the opaque classes (opaque,
 *      two-sided) also in the depth pre-pass (depth only, the same vertex shader with invariant gl_Position), then in
 *      the scene pass with EQUAL and no depth write; the PrototypeFoliage class (prototype_foliage.vert/.frag, alpha
 *      to coverage, the foliage texture) is never in the pre-pass. The draw pipelines have a second descriptor set
 *      (per frame slot): set 1 of SceneLightingSet (S36: the lighting block, the foliage texture, the lighting's
 *      resources); every scene pipeline lights with shadeSurface (S36, lighting.glsl).
 * S36a2 (§10.2): views 1..4 are the shadow cascades' light-space volumes (ShadowView): each set's bounds are tested
 * against every view on the CPU and its work chunks carry a view mask (WorkChunk::pad, 16 bits; scratchBase, the
 * prefix sum of 256 * popcount(mask), in a parallel buffer); the count and the write pass loop over the chunk's views
 * (one scratch entry per instance and view; shadow views: the volume, the de-duplication by the previous cascade's xy
 * rectangle, the small-caster rule, the LOD of the main camera's distance plus the view's bias; the main camera's end
 * distances apply to every view); buckets, commands and the scan grow by the view count (view * bucketsPerView + ...).
 * Shared device buffers (counts, scratch, compacted, commands, bucket LODs) are recreated at frame start when too
 * small; the per-frame host visible buffers (offset table, work chunks, dynamics, statistics, staging) belong to a
 * frame slot (its previous frame has completed). One descriptor set per frame slot with every binding, rewritten for
 * the frame being recorded. Every storage buffer bound whole stays within storageLimit() (maxStorageBufferRange);
 * more is a fatal error. The draws use the frame's snapshot of the bucket layout and the vertex and index buffers
 * (taken in recordTransfer): content changes in the render callback take effect in the next frame.
 */
class SAIGA_VULKAN_API SceneCulling
{
   public:
    // lighting: set 1 of the draws (RenderScene's, outlives this)
    void init(VulkanBase& base, const RenderingFormats& formats, const RenderSceneParameters& params,
              SceneLightingSet& lighting);
    // The size limit of a storage buffer bound whole (maxStorageBufferRange or the parameters' lower test limit)
    uint64_t storageLimit() const { return storageBytes; }
    void destroy();

    // The pipelines created since init (tests: a debug view switch creates none)
    uint64_t pipelineCreations() const { return creations; }

    // The transfer section: growth copies, uploads (meshes, instances, tables, the commands' static fields), the
    // frame's offset table, work chunks and dynamics, the buffer sizes and the frame's descriptor set
    // shadowViews (S36a2): the views 1.. of this frame (at most kMaxCullViews - 1)
    void recordTransfer(vk::CommandBuffer cmd, uint32_t frameIndex, MeshPool& meshes, InstanceBuffer& instances,
                        const SceneCamera& camera, const std::vector<ShadowView>& shadowViews = {});
    // The cull section (between transfer and scene): fill, count, scan, write
    void recordCull(vk::CommandBuffer cmd, uint32_t frameIndex, MeshPool& meshes, const SceneCamera& camera);
    // Inside the depth pre-pass (depth only): the opaque classes; returns the number of indirect draw calls
    uint32_t recordPrepass(vk::CommandBuffer cmd, uint32_t frameIndex, const SceneCamera& camera);
    // Inside the scene pass, from the frame's snapshot (not the mesh pool: the render callback may have changed it);
    // prepass: the opaque classes were drawn in the pre-pass (EQUAL, no depth write); returns the number of indirect
    // draw calls
    uint32_t recordDraw(vk::CommandBuffer cmd, uint32_t frameIndex, const SceneCamera& camera, bool prepass);
    // S36a2: inside a cascade's depth pass, the casters of view (1..) with the classes' shadow pipelines (a class
    // without an entry casts nothing; the PrototypeFoliage class only with the foliage texture); returns the number of
    // indirect draw calls
    uint32_t recordShadow(vk::CommandBuffer cmd, uint32_t frameIndex, uint32_t view, const mat4& viewProj, float alpha,
                          ShadowPipelineTable& table, float slopeBias);
    // S36b: inside an atlas face's pass (D16), the casters of view with the classes' atlas pipelines (the linear depth;
    // light: xyz camera-relative, w 1 / range)
    uint32_t recordAtlas(vk::CommandBuffer cmd, uint32_t frameIndex, uint32_t view, const mat4& viewProj, float alpha,
                         ShadowPipelineTable& table, const vec4& light);
    // The views of the frame (1 + its shadow views; 0 without a cull)
    uint32_t viewCount(uint32_t frameIndex) const { return frames[frameIndex].viewCount; }
    // The scene's set-0 layout (the shadow pipelines are created with it)
    static DescriptorSetLayout setLayout();
    vk::DescriptorSet sceneSet(uint32_t frameIndex) const { return frames[frameIndex].set; }
    // The foliage texture of the PrototypeFoliage class exists (RenderScene::setFoliageTexture; binding 1 of the
    // lighting set): without it the class is not drawn
    void setFoliageTexture(Texture2DArray* texture) { foliageTexture = texture; }

    // The statistics the scan wrote into the frame slot's readback (instancesPerLod, nonEmptyBuckets, the tracked
    // meshes); zeros if the frame did not cull
    void readStatistics(uint32_t frameIndex, RenderStatistics& out);
    // The frame culled (work chunks and a camera)
    bool culled(uint32_t frameIndex) const { return frames[frameIndex].culled; }
    bool multiDrawIndirect() const { return multiDraw; }
    // The tracked meshes of the statistics (RenderScene::trackMeshStatistics), from the next recordTransfer
    void setTrackedMeshes(const std::vector<MeshId>& meshes);

   private:
    struct Frame
    {
        Buffer offsets, chunks, dynamics, stats, staging, views, scratchBases;
        vk::DescriptorSet set;
        uint32_t chunkCount = 0;
        bool culled         = false;
        // S36a2: the views of the frame and the buckets per view of its snapshot
        uint32_t viewCount      = 1;
        uint32_t bucketsPerView = 0;
        // The snapshot of recordTransfer for the draws: each pipeline class's buckets, the buffers
        struct Range
        {
            uint32_t first = 0, count = 0;
        };
        std::array<Range, pipelineClassCount> ranges = {};
        vk::Buffer vertices, indices, commands;
    };
    struct WorkChunk
    {
        uint32_t first, count, slot, pad;
    };

    VulkanBase* base              = nullptr;
    SceneLightingSet* lightingSet = nullptr;
    RenderingFormats sceneFormats;
    bool multiDraw     = false;
    uint64_t creations = 0;
    // RenderSceneParameters::dynamicCapacity: the initial size of the dynamics and compacted buffers
    uint32_t dynamicCapacity      = 0;
    uint32_t maxDrawIndirectCount = 1;
    uint32_t maxWorkGroups        = 65535;
    uint32_t maxWorkGroupsY       = 65535;
    uint64_t storageBytes         = 0;
    ComputePipeline countPipeline, scanPipeline, writePipeline;
    // Per class (S36: one light model; the camera-relative position is always a varying: the fog, the cloud shadow and
    // the specular of every fragment need it, so the SCENE_FOG variant of S19a is gone)
    std::array<Pipeline, pipelineClassCount> drawPipelines;
    // The opaque classes (opaque, two-sided): depth only for the pre-pass, EQUAL without depth write after it
    std::array<Pipeline, 2> prepassPipelines;
    std::array<Pipeline, 2> equalPipelines;
    void createDrawPipelines();
    void destroyDrawPipelines();
    Texture2DArray* foliageTexture = nullptr;
    std::array<Frame, 2> frames;
    Buffer counts, scratch, compacted, commands, bucketLods;
    uint64_t uploadedLayout = 0;
    uint32_t bucketCount    = 0;
    // The view count the commands' static fields were uploaded for (S36a2)
    uint32_t uploadedViews = 0;
    std::vector<uint32_t> scratchBaseList;
    UploadBatch batch;
    std::vector<WorkChunk> chunkList;
    std::vector<MeshId> trackedMeshes;

    // The new size of a storage buffer: max(2x, required) within the limit; more than the limit is a fatal error
    size_t storageSize(const Buffer& b, size_t bytes, const char* what) const;
    void ensureHost(Buffer& b, size_t bytes, vk::BufferUsageFlags usage, const char* what);
    void ensureDevice(Buffer& b, size_t bytes, vk::BufferUsageFlags usage, bool& recreated, const char* what);
    void writeDescriptors(Frame& f, MeshPool& meshes, InstanceBuffer& instances);
    // One workgroup per chunk, 2D beyond maxWorkGroups
    void dispatchChunks(vk::CommandBuffer cmd, uint32_t chunkCount) const;
    // The indirect draws of one class range with the pipeline (bound with the frame's sets and push constants)
    uint32_t drawClass(vk::CommandBuffer cmd, uint32_t frameIndex, int klass, Pipeline& p, const SceneCamera& camera);
    // The indirect draws of one class range of a view (the pipeline bound with its sets and push constants)
    uint32_t drawRange(vk::CommandBuffer cmd, uint32_t frameIndex, int klass, uint32_t view);
};

}  // namespace Vulkan
}  // namespace Saiga
