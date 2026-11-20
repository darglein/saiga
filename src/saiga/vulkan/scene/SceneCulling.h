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

/**
 * The GPU culling and the instance draws of RenderScene (S18, docs/engine_design.md §10.2):
 *   1. CPU: the instance sets' bounds against the frustum and their meshes' end distances -> work chunks of at most 256
 *      instances {first, count, slot} over each visible set's used size; the dynamics chunked with slot 0xFFFFFFFF;
 *   2. count pass (instance_count.comp, one workgroup per chunk): rejections, frustum, LOD, atomicAdd into the bucket's
 *      count; the bucket and the returned slot are stored in the scratch entry;
 *   3. scan (bucket_scan.comp): instanceCount and firstInstance of every bucket's indirect command, the statistics;
 *   4. write pass (instance_write.comp): the compacted entries uvec2(instance index | dynamic bit, slot | LOD << 24);
 *   5. draw: one vkCmdDrawIndexedIndirect per pipeline class and view over its buckets (split at
 *      maxDrawIndirectCount), or one per bucket without multiDrawIndirect.
 * Shared device buffers (counts, scratch, compacted, commands, bucket LODs) are recreated at frame start when too
 * small; the per-frame host visible buffers (offset table, work chunks, dynamics, statistics, staging) belong to a
 * frame slot (its previous frame has completed). One descriptor set per frame slot with every binding, rewritten for
 * the frame being recorded.
 */
class SAIGA_VULKAN_API SceneCulling
{
   public:
    void init(VulkanBase& base, const RenderingFormats& formats, const RenderSceneParameters& params);
    void destroy();

    // The transfer section: growth copies, uploads (meshes, instances, tables, the commands' static fields), the
    // frame's offset table, work chunks and dynamics, the buffer sizes and the frame's descriptor set
    void recordTransfer(vk::CommandBuffer cmd, uint32_t frameIndex, MeshPool& meshes, InstanceBuffer& instances,
                        const SceneCamera& camera);
    // The cull section (between transfer and scene): fill, count, scan, write
    void recordCull(vk::CommandBuffer cmd, uint32_t frameIndex, MeshPool& meshes, const SceneCamera& camera);
    // Inside the scene pass; returns the number of indirect draw calls
    uint32_t recordDraw(vk::CommandBuffer cmd, uint32_t frameIndex, MeshPool& meshes, const SceneCamera& camera);

    // The statistics the scan wrote into the frame slot's readback (instancesPerLod, nonEmptyBuckets); zeros if the
    // frame did not cull
    void readStatistics(uint32_t frameIndex, RenderStatistics& out);
    // The frame culled (work chunks and a camera)
    bool culled(uint32_t frameIndex) const { return frames[frameIndex].culled; }
    bool multiDrawIndirect() const { return multiDraw; }

   private:
    struct Frame
    {
        Buffer offsets, chunks, dynamics, stats, staging;
        vk::DescriptorSet set;
        uint32_t chunkCount = 0;
        bool culled         = false;
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
    ComputePipeline countPipeline, scanPipeline, writePipeline;
    std::array<Pipeline, pipelineClassCount> drawPipelines;
    std::array<Frame, 2> frames;
    Buffer counts, scratch, compacted, commands, bucketLods;
    uint64_t uploadedLayout = 0;
    uint32_t bucketCount    = 0;
    UploadBatch batch;
    std::vector<WorkChunk> chunkList;

    void ensureHost(Buffer& b, size_t bytes, vk::BufferUsageFlags usage);
    void ensureDevice(Buffer& b, size_t bytes, vk::BufferUsageFlags usage, bool& recreated);
    void writeDescriptors(Frame& f, MeshPool& meshes, InstanceBuffer& instances);
};

}  // namespace Vulkan
}  // namespace Saiga
