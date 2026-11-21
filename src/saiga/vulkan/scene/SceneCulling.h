/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"
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

// The lighting block of the instance draws (std140, shader/vulkan/scene/scene_lighting.glsl; S19a, a RenderScene block
// until S36's lighting): the defaults are S18's fixed light (sun normalize(0.4, 1, 0.7) with 0.8, ambient 0.35, no
// hemisphere term, fog off); RenderScene::setPrototypeEnvironment takes the prototype's sun, ambient, fog and wind
struct SceneLighting
{
    // xyz: towards the sun (normalized), w: intensity
    vec4 sunDir = vec4(0.3113996f, 0.7784989f, 0.5449493f, 0.8f);
    // x: ambient, y: 1 hemisphere ambient (0.6 + 0.4 n.y, the prototype's) else constant, z: albedo (foliage clay), w:
    // the foliage's alpha cutoff
    vec4 ambient = vec4(0.35f, 0, 0.55f, 0.45f);
    // x: density, y: height falloff, z: enabled (0/1), w: the fog's base height (the water level)
    vec4 fog      = vec4(0, 0, 0, 0);
    vec4 fogColor = vec4(0.42f, 0.42f, 0.42f, 1);
    // xyz: the camera position (float, the space's coordinates: the foliage's wind phase and the fog height), w: the
    // MSAA samples
    vec4 camera = vec4(0, 0, 0, 1);
    // xyz: the tree wind phases fmod(time * (1.1, 0.9, 4.0), 2 pi) (computed in double), w: the wind strength
    vec4 wind = vec4(0, 0, 0, 0);
    // x: debug mode (1: tint by LOD)
    vec4 debug = vec4(0, 0, 0, 0);
};
static_assert(sizeof(SceneLighting) == 112, "SceneLighting of scene_lighting.glsl");

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
 *      (per frame slot): the lighting block (SceneLighting) and the foliage texture.
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
    void init(VulkanBase& base, const RenderingFormats& formats, const RenderSceneParameters& params);
    // The size limit of a storage buffer bound whole (maxStorageBufferRange or the parameters' lower test limit)
    uint64_t storageLimit() const { return storageBytes; }
    void destroy();

    // The transfer section: growth copies, uploads (meshes, instances, tables, the commands' static fields), the
    // frame's offset table, work chunks and dynamics, the buffer sizes and the frame's descriptor set
    void recordTransfer(vk::CommandBuffer cmd, uint32_t frameIndex, MeshPool& meshes, InstanceBuffer& instances,
                        const SceneCamera& camera, const SceneLighting& lighting);
    // The cull section (between transfer and scene): fill, count, scan, write
    void recordCull(vk::CommandBuffer cmd, uint32_t frameIndex, MeshPool& meshes, const SceneCamera& camera);
    // Inside the depth pre-pass (depth only): the opaque classes; returns the number of indirect draw calls
    uint32_t recordPrepass(vk::CommandBuffer cmd, uint32_t frameIndex, const SceneCamera& camera);
    // Inside the scene pass, from the frame's snapshot (not the mesh pool: the render callback may have changed it);
    // prepass: the opaque classes were drawn in the pre-pass (EQUAL, no depth write); returns the number of indirect
    // draw calls
    uint32_t recordDraw(vk::CommandBuffer cmd, uint32_t frameIndex, const SceneCamera& camera, bool prepass);
    // The foliage texture of the PrototypeFoliage class (RenderScene::setPrototypeEnvironment; nullptr: none), written
    // into the frame slots' sets from the next recordTransfer
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
        Buffer offsets, chunks, dynamics, stats, staging, lighting;
        vk::DescriptorSet set, lightingSet;
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

    VulkanBase* base = nullptr;
    bool multiDraw   = false;
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
    uint32_t drawClass(vk::CommandBuffer cmd, Frame& f, int klass, Pipeline& p, const SceneCamera& camera);
};

}  // namespace Vulkan
}  // namespace Saiga
