/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/scene/RenderSceneTypes.h"
#include "saiga/vulkan/scene/SubAllocatedBuffer.h"

#include <map>
#include <utility>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class MeshPool;
class UploadBatch;

/**
 * The instances of RenderScene (S18, docs/engine_design.md §10.1):
 * - static instance sets: each a contiguous range of `capacity` StaticInstances in one device buffer
 *   (SubAllocatedBuffer), with its region, its bounds (an AABB around its instances' bounding spheres, region-relative,
 *   computed on the CPU at add and update; tombstones keep the bounds) and a stable offset-table slot. The CPU copy of
 *   each set is the source of the uploads, which are queued by set and range (merged per frame);
 * - dynamic instances: the CPU array is the source of truth, written into the frame's dynamics buffer every frame; each
 *   holds its chunk's offset slot (a reference-counted ChunkId -> slot map);
 * - the offset-table slots of the sets (regions) and of the dynamics' chunks.
 */
class SAIGA_VULKAN_API InstanceBuffer
{
   public:
    struct Set
    {
        RegionId region;
        uint32_t capacity = 0;
        // The highest written index + 1: the culling visits [0, used)
        uint32_t used = 0;
        std::vector<StaticInstance> instances;
        // Region-relative AABB of the bounding spheres without the meshes' bounds padding, the largest instance scale
        // and the meshes of the instances (sorted, distinct): the padded bounds follow a padding change
        // (RenderScene::setMeshLodDistances) without visiting the instances
        Vec3 spheresMin = Vec3::Zero(), spheresMax = Vec3::Zero();
        double maxScale = 0;
        std::vector<MeshId> meshes;
        // Region-relative AABB of the padded bounding spheres: the unpadded AABB grown by the largest padding of its
        // meshes times maxScale (empty: hasBounds false)
        Vec3 boundsMin = Vec3::Zero(), boundsMax = Vec3::Zero();
        bool hasBounds = false;
        // The largest last end distance of its instances' meshes
        double maxEndDistance = 0;
        uint32_t slot         = 0;
        // Ranges [first, first + count) to upload
        std::vector<std::pair<uint32_t, uint32_t>> dirty;
    };
    struct Dynamic
    {
        bool alive = false, set = false;
        MeshId mesh = 0;
        ChunkId chunk;
        uint32_t slot = 0;
        DynamicPose prev, curr;
        uint32_t boneOffset = 0;
    };
    struct Slot
    {
        enum class Kind : uint8_t
        {
            Free,
            Region,
            Chunk
        };
        Kind kind = Kind::Free;
        RegionId region;
        ChunkId chunk;
    };
    // shader/vulkan/scene/instance_common.glsl: DynamicInstance (80 bytes)
    struct GpuDynamic
    {
        float prevPos[4] = {0, 0, 0, 0};
        float currPos[4] = {0, 0, 0, 0};
        float prevRot[4] = {0, 0, 0, 1};
        float currRot[4] = {0, 0, 0, 1};
        uint32_t mesh = 0, slot = 0, boneOffset = 0, pad = 0;
    };
    static_assert(sizeof(GpuDynamic) == 80, "DynamicInstance of instance_common.glsl");

    // maxBytes: the limit of the instance buffer (maxStorageBufferRange; growth beyond it is a fatal error)
    void init(VulkanBase& base, uint32_t staticCapacity, uint64_t maxBytes);
    void destroy();

    InstanceSetHandle addSet(const RegionId& region, const std::vector<StaticInstance>& instances, uint32_t capacity,
                             const MeshPool& meshes);
    void update(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& instances,
                const MeshPool& meshes);
    void removeInstance(InstanceSetHandle set, uint32_t index);
    void removeSet(InstanceSetHandle set);
    // The padded bounds and end distances of the sets with instances of the mesh (after setMeshLodDistances)
    void refreshBounds(MeshId mesh, const MeshPool& meshes);

    InstanceHandle addDynamic(MeshId mesh);
    void setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev, const DynamicPose& curr,
                    uint32_t boneOffset);
    void removeDynamic(InstanceHandle h);

    const std::map<uint32_t, Set>& sets() const { return setMap; }
    // The first instance of the set in the buffer
    uint32_t setOffset(uint32_t id) const { return uint32_t(buffer.offset(id)); }
    const std::vector<Dynamic>& dynamics() const { return dynamicList; }
    const std::vector<Slot>& slots() const { return slotList; }
    // The capacity of the static instance buffer (instances)
    uint64_t capacity() const { return buffer.capacity(); }

    bool recordGrowth(vk::CommandBuffer cmd) { return buffer.recordGrowth(cmd); }
    void queueUploads(UploadBatch& batch);
    Buffer& instanceBuffer() { return buffer.buffer(); }
    // Writes dynamics().size() entries (free or unset entries with mesh 0)
    void writeDynamics(GpuDynamic* out) const;

   private:
    SubAllocatedBuffer buffer;
    std::map<uint32_t, Set> setMap;
    uint32_t nextSet = 1;
    std::vector<Dynamic> dynamicList;  // InstanceHandle id - 1
    std::vector<uint32_t> freeDynamics;
    std::vector<Slot> slotList;
    std::vector<uint32_t> freeSlots;
    std::map<ChunkId, std::pair<uint32_t, uint32_t>> chunkSlots;  // slot, references

    uint32_t allocateSlot(const Slot& s);
    void freeSlot(uint32_t slot);
    uint32_t acquireChunk(const ChunkId& chunk);
    void releaseChunk(const ChunkId& chunk);
    void computeBounds(Set& s, const MeshPool& meshes);
    void padBounds(Set& s, const MeshPool& meshes);
};

}  // namespace Vulkan
}  // namespace Saiga
