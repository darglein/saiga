/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/scene/RenderSceneTypes.h"
#include "saiga/vulkan/scene/SubAllocatedBuffer.h"

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class UploadBatch;

/**
 * The meshes and materials of RenderScene (S18, docs/engine_design.md §10.1, §10.8): one vertex and one index buffer
 * sub-allocated per mesh (SubAllocatedBuffer, element units), the GPU mesh record table (MeshId -> record, sized to the
 * highest registered ID; zeroed records for unregistered IDs), the material buffer (MaterialId -> material) and the
 * palette, and the bucket layout of the culling.
 *
 * Buckets are dense: one per existing LOD of a registered mesh (per view), sorted by (pipeline class, mesh registration
 * order, LOD), so each pipeline class's buckets are a contiguous range. addMesh, a class change of setMaterial and a
 * growth of the vertex or index buffer change the layout (layoutVersion): the mesh records (bucketBase, offsets) and
 * the commands' static fields are rewritten through transfer commands (never a host mapping: frames in flight read
 * them).
 */
class SAIGA_VULKAN_API MeshPool
{
   public:
    struct Lod
    {
        // Relative to the mesh's ranges
        uint32_t firstIndex = 0, indexCount = 0;
        uint32_t firstVertex = 0;
        float endDistance    = 0;
    };
    struct Mesh
    {
        std::string name;
        MaterialId material = 0;
        uint32_t order      = 0;  // registration order
        vec3 sphereCenter   = vec3(0, 0, 0);
        float sphereRadius = 0, boundsPadding = 0;
        std::vector<Lod> lods;
        uint32_t vertexCount = 0, indexCount = 0;
        // The data until its upload is recorded
        std::vector<MeshVertex> pendingVertices;
        std::vector<uint32_t> pendingIndices;
        float lastEndDistance() const { return lods.back().endDistance; }
    };
    struct Bucket
    {
        MeshId mesh   = 0;
        uint8_t lod   = 0;
        uint8_t klass = 0;  // pipeline class index 0..pipelineClassCount-1
    };
    struct ClassRange
    {
        uint32_t first = 0, count = 0;
    };

    void init(VulkanBase& base, uint32_t vertexCapacity, uint32_t indexCapacity);
    void destroy();

    void addMaterial(MaterialId id, const std::string& name, const MaterialData& data);
    void setMaterial(MaterialId id, const MaterialData& data);
    bool hasMaterial(MaterialId id) const { return id != 0 && id < materials.size() && materials[id].registered; }
    void addMesh(MeshId id, const std::string& name, const MeshLods& lods, MaterialId material);
    // Rewrites the mesh record's end distances (same LOD count, non-decreasing, >= 0) and bounds padding: no geometry
    // upload, no bucket layout change (RenderScene::setMeshLodDistances)
    void setLodDistances(MeshId id, const std::vector<float>& endDistances, float boundsPadding);
    void setPalette(const std::vector<PaletteEntry>& palette);

    // nullptr if not registered
    const Mesh* mesh(MeshId id) const;
    // The ID of a registered name (0: none)
    MeshId findMesh(const std::string& name) const
    {
        auto it = meshNames.find(name);
        return it == meshNames.end() ? MeshId(0) : it->second;
    }
    MaterialId findMaterial(const std::string& name) const
    {
        auto it = materialNames.find(name);
        return it == materialNames.end() ? MaterialId(0) : it->second;
    }
    // Highest registered MeshId + 1
    uint32_t meshTableSize() const { return uint32_t(meshes.size()); }
    uint32_t meshCount() const { return registeredMeshes; }

    // The pipeline class index (0..pipelineClassCount-1) of a material
    static uint8_t pipelineClassIndex(const MaterialData& m);
    // The pipeline class index of a registered mesh's material
    uint8_t meshClass(MeshId id) const { return pipelineClassIndex(materials[meshes[id].material].data); }

    uint64_t layoutVersion() const { return version; }
    // Per view (S18: one view)
    const std::vector<Bucket>& buckets() const { return bucketList; }
    const std::array<ClassRange, pipelineClassCount>& classRanges() const { return ranges; }
    // The commands' static fields of a bucket (instanceCount and firstInstance 0)
    vk::DrawIndexedIndirectCommand command(uint32_t bucket) const;

    // Records the growth copies and queues the uploads (vertex and index data, mesh records, materials, palette);
    // recreates the shared tables (mesh records, materials, palette) when they are too small (the old ones through
    // deferDelete). Returns true if a growth copy was recorded (UploadBatch records the uploads after a barrier).
    bool recordGrowth(vk::CommandBuffer cmd);
    void queueUploads(UploadBatch& batch);

    Buffer& vertexBuffer() { return vertices.buffer(); }
    Buffer& indexBuffer() { return indices.buffer(); }
    Buffer& meshRecordBuffer() { return meshRecords; }
    Buffer& materialBuffer() { return materialTable; }
    Buffer& paletteBuffer() { return paletteTable; }
    // The entries of the palette buffer (at least 1: a white entry without a palette)
    uint32_t paletteCount() const { return uint32_t(std::max<size_t>(paletteEntries.size(), 1)); }

   private:
    struct MaterialEntry
    {
        bool registered = false;
        std::string name;
        MaterialData data;
    };

    VulkanBase* base = nullptr;
    SubAllocatedBuffer vertices, indices;
    std::vector<Mesh> meshes;  // by MeshId; registered: !lods.empty()
    std::vector<MeshId> order;
    std::map<std::string, MeshId> meshNames;
    uint32_t registeredMeshes = 0;
    std::vector<MaterialEntry> materials;  // by MaterialId
    std::map<std::string, MaterialId> materialNames;
    std::vector<PaletteEntry> paletteEntries;

    std::vector<Bucket> bucketList;
    std::array<ClassRange, pipelineClassCount> ranges = {};
    std::vector<uint32_t> bucketBase;  // by MeshId
    uint64_t version          = 1;
    uint64_t vertexGeneration = 0, indexGeneration = 0;
    bool recordsDirty = true, materialsDirty = true, paletteDirty = true;
    Buffer meshRecords, materialTable, paletteTable;

    void rebuildLayout();
    void checkGrowth();
};

}  // namespace Vulkan
}  // namespace Saiga
