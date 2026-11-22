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
 *      resources). S36a1b: two pipeline variants, the legacy S19a light model (with its SCENE_FOG variant) and the
 *      LIGHTING_PBR one (LightingParameters::pbr; created on demand by setPbr, the other variant's pipelines released).
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

    // The draw pipelines of the light model (LightingParameters::pbr; at a frame start, outside of the recording): a
    // change creates the variant's pipelines and releases the others (deferred)
    void setPbr(bool pbr);
    bool pbr() const { return pbrVariant; }
    // The pipelines created since init (tests: a debug view switch creates none)
    uint64_t pipelineCreations() const { return creations; }

    // The transfer section: growth copies, uploads (meshes, instances, tables, the commands' static fields), the
    // frame's offset table, work chunks and dynamics, the buffer sizes and the frame's descriptor set. fog: the legacy
    // variant's SCENE_FOG pipelines (the lighting block's fog is enabled)
    void recordTransfer(vk::CommandBuffer cmd, uint32_t frameIndex, MeshPool& meshes, InstanceBuffer& instances,
                        const SceneCamera& camera, bool fog);
    // The cull section (between transfer and scene): fill, count, scan, write
    void recordCull(vk::CommandBuffer cmd, uint32_t frameIndex, MeshPool& meshes, const SceneCamera& camera);
    // Inside the depth pre-pass (depth only): the opaque classes; returns the number of indirect draw calls
    uint32_t recordPrepass(vk::CommandBuffer cmd, uint32_t frameIndex, const SceneCamera& camera);
    // Inside the scene pass, from the frame's snapshot (not the mesh pool: the render callback may have changed it);
    // prepass: the opaque classes were drawn in the pre-pass (EQUAL, no depth write); returns the number of indirect
    // draw calls
    uint32_t recordDraw(vk::CommandBuffer cmd, uint32_t frameIndex, const SceneCamera& camera, bool prepass);
    // The foliage texture of the PrototypeFoliage class exists (RenderScene::setPrototypeEnvironment; binding 1 of the
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
        Buffer offsets, chunks, dynamics, stats, staging;
        vk::DescriptorSet set;
        uint32_t chunkCount = 0;
        bool culled         = false;
        // The snapshot of recordTransfer for the draws: each pipeline class's buckets, the buffers
        struct Range
        {
            uint32_t first = 0, count = 0;
        };
        std::array<Range, pipelineClassCount> ranges = {};
        vk::Buffer vertices, indices, commands;
        // The lighting block's fog is enabled: the SCENE_FOG variant of the flat pipelines
        bool fog = false;
    };
    struct WorkChunk
    {
        uint32_t first, count, slot, pad;
    };

    VulkanBase* base              = nullptr;
    SceneLightingSet* lightingSet = nullptr;
    RenderingFormats sceneFormats;
    bool multiDraw     = false;
    bool pbrVariant    = false;
    uint64_t creations = 0;
    // RenderSceneParameters::dynamicCapacity: the initial size of the dynamics and compacted buffers
    uint32_t dynamicCapacity      = 0;
    uint32_t maxDrawIndirectCount = 1;
    uint32_t maxWorkGroups        = 65535;
    uint32_t maxWorkGroupsY       = 65535;
    uint64_t storageBytes         = 0;
    ComputePipeline countPipeline, scanPipeline, writePipeline;
    // [fog variant][class]: the flat classes without (0) and with (1) the height fog (SCENE_FOG: the camera-relative
    // position as a varying, which costs ~7 % of the flat draws when unused); the foliage class always has fog
    std::array<std::array<Pipeline, pipelineClassCount>, 2> drawPipelines;
    // The opaque classes (opaque, two-sided): depth only for the pre-pass, EQUAL without depth write after it ([fog
    // variant][class])
    std::array<Pipeline, 2> prepassPipelines;
    std::array<std::array<Pipeline, 2>, 2> equalPipelines;
    // The LIGHTING_PBR variant (S36a1b): every class and the opaque classes after the pre-pass; the relative position
    // is always a varying (no fog variant)
    std::array<Pipeline, pipelineClassCount> pbrPipelines;
    std::array<Pipeline, 2> pbrEqualPipelines;
    void createDrawPipelines();
    void destroyDrawPipelines();
    Texture2DArray* foliageTexture = nullptr;
    std::array<Frame, 2> frames;
    Buffer counts, scratch, compacted, commands, bucketLods;
    uint64_t uploadedLayout = 0;
    uint32_t bucketCount    = 0;
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
};

}  // namespace Vulkan
}  // namespace Saiga
