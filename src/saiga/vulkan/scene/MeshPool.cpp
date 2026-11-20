/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/MeshPool.h"

#include "saiga/vulkan/scene/UploadBatch.h"

#include <algorithm>

namespace Saiga
{
namespace Vulkan
{
namespace
{
// shader/vulkan/scene/instance_common.glsl: MeshRecord (112 bytes)
struct GpuMeshLod
{
    uint32_t firstIndex  = 0;
    uint32_t indexCount  = 0;
    int32_t vertexOffset = 0;
    float endDistance    = 0;
};
struct GpuMeshRecord
{
    uint32_t material      = 0;
    uint32_t lodCount      = 0;
    uint32_t bucketBase    = 0;
    uint32_t pipelineClass = 0;
    float sphere[4]        = {0, 0, 0, 0};
    float padding[4]       = {0, 0, 0, 0};
    GpuMeshLod lods[4];
};
static_assert(sizeof(GpuMeshRecord) == 112, "MeshRecord of instance_common.glsl");

// instance_common.glsl / flat.frag: Material (48 bytes)
struct GpuMaterial
{
    float baseColor[4]         = {1, 1, 1, 1};
    float emissiveRoughness[4] = {0, 0, 0, 0.8f};
    uint32_t model             = 0;
    uint32_t flags             = 0;
    float metalness            = 0;
    uint32_t pad               = 0;
};
static_assert(sizeof(GpuMaterial) == 48, "Material of flat.frag");

GpuMaterial toGpu(const MaterialData& m)
{
    GpuMaterial g;
    for (int i = 0; i < 4; ++i) g.baseColor[i] = m.baseColor(i);
    for (int i = 0; i < 3; ++i) g.emissiveRoughness[i] = m.emissive(i);
    g.emissiveRoughness[3] = m.roughness;
    g.model                = m.model == ShadingModel::SmoothTextured ? 1 : 0;
    g.flags                = (m.usePalette ? 1u : 0u) | (m.alphaMask ? 2u : 0u) | (m.twoSided ? 4u : 0u);
    g.metalness            = m.metalness;
    return g;
}

const vk::BufferUsageFlags tableUsage = vk::BufferUsageFlagBits::eStorageBuffer |
                                        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;

// Recreates the table when it is smaller than size bytes (max(2x, size); the old buffer through deferDelete)
void ensureTable(VulkanBase& base, Buffer& table, size_t size)
{
    if (table && table.size() >= size) return;
    size_t capacity = std::max<size_t>(size, size_t(table.size()) * 2);
    table.createBuffer(base, std::max<size_t>(capacity, 256), tableUsage, vk::MemoryPropertyFlagBits::eDeviceLocal);
}
}  // namespace

void MeshPool::init(VulkanBase& b, uint32_t vertexCapacity, uint32_t indexCapacity)
{
    destroy();
    base = &b;
    vertices.init(b, sizeof(MeshVertex), vertexCapacity, vk::BufferUsageFlagBits::eVertexBuffer);
    indices.init(b, sizeof(uint32_t), indexCapacity, vk::BufferUsageFlagBits::eIndexBuffer);
    meshes.resize(1);
    materials.resize(1);
    vertexGeneration = vertices.generation();
    indexGeneration  = indices.generation();
    recordsDirty = materialsDirty = paletteDirty = true;
    rebuildLayout();
}

void MeshPool::destroy()
{
    vertices.destroy();
    indices.destroy();
    meshRecords.destroy();
    materialTable.destroy();
    paletteTable.destroy();
    meshes.clear();
    order.clear();
    meshNames.clear();
    materials.clear();
    materialNames.clear();
    paletteEntries.clear();
    bucketList.clear();
    bucketBase.clear();
    registeredMeshes = 0;
}

uint8_t MeshPool::pipelineClassIndex(const MaterialData& m)
{
    if (m.pipelineClass != 0)
    {
        SAIGA_ASSERT(m.pipelineClass <= pipelineClassCount, "MaterialData::pipelineClass: unknown class");
        return uint8_t(m.pipelineClass - 1);
    }
    return uint8_t((m.alphaMask ? 1 : 0) + (m.twoSided ? 2 : 0));
}

void MeshPool::addMaterial(MaterialId id, const std::string& name, const MaterialData& data)
{
    SAIGA_ASSERT(id != 0, "addMaterial: MaterialId 0 is none");
    SAIGA_ASSERT(id >= materials.size() || !materials[id].registered, "addMaterial: the MaterialId is registered");
    SAIGA_ASSERT(materialNames.count(name) == 0, "addMaterial: the material name is registered");
    pipelineClassIndex(data);
    if (id >= materials.size()) materials.resize(size_t(id) + 1);
    materials[id]       = MaterialEntry{true, name, data};
    materialNames[name] = id;
    materialsDirty      = true;
}

void MeshPool::setMaterial(MaterialId id, const MaterialData& data)
{
    SAIGA_ASSERT(hasMaterial(id), "setMaterial: the material is not registered");
    const bool classChanged = pipelineClassIndex(materials[id].data) != pipelineClassIndex(data);
    materials[id].data      = data;
    materialsDirty          = true;
    if (classChanged) rebuildLayout();
}

void MeshPool::setPalette(const std::vector<PaletteEntry>& palette)
{
    SAIGA_ASSERT(palette.size() <= 4096, "setPalette: at most 4096 entries");
    paletteEntries = palette;
    paletteDirty   = true;
}

const MeshPool::Mesh* MeshPool::mesh(MeshId id) const
{
    if (id == 0 || id >= meshes.size() || meshes[id].lods.empty()) return nullptr;
    return &meshes[id];
}

void MeshPool::addMesh(MeshId id, const std::string& name, const MeshLods& lods, MaterialId material)
{
    SAIGA_ASSERT(id != 0, "addMesh: MeshId 0 is none");
    SAIGA_ASSERT(!mesh(id), "addMesh: the MeshId is registered");
    SAIGA_ASSERT(meshNames.count(name) == 0, "addMesh: the mesh name is registered");
    SAIGA_ASSERT(hasMaterial(material), "addMesh: the material is not registered");
    SAIGA_ASSERT(!lods.lods.empty() && lods.lods.size() <= 4, "addMesh: 1 to 4 LODs");
    Mesh m;
    m.name          = name;
    m.material      = material;
    m.order         = uint32_t(order.size());
    m.sphereCenter  = lods.sphereCenter;
    m.sphereRadius  = lods.sphereRadius;
    m.boundsPadding = lods.boundsPadding;
    float previous  = 0;
    for (const MeshLod& l : lods.lods)
    {
        // Non-decreasing: an end distance equal to the previous one is an empty LOD (never selected)
        SAIGA_ASSERT(l.endDistance >= previous, "addMesh: the LODs' end distances must be >= 0 and non-decreasing");
        SAIGA_ASSERT(!l.indices.empty() && l.indices.size() % 3 == 0, "addMesh: a LOD needs triangles");
        previous = l.endDistance;
        Lod lod;
        lod.firstIndex  = m.indexCount;
        lod.indexCount  = uint32_t(l.indices.size());
        lod.firstVertex = m.vertexCount;
        lod.endDistance = l.endDistance;
        for (uint32_t i : l.indices) SAIGA_ASSERT(i < l.vertices.size(), "addMesh: index out of range");
        m.pendingVertices.insert(m.pendingVertices.end(), l.vertices.begin(), l.vertices.end());
        m.pendingIndices.insert(m.pendingIndices.end(), l.indices.begin(), l.indices.end());
        m.vertexCount += uint32_t(l.vertices.size());
        m.indexCount += lod.indexCount;
        m.lods.push_back(lod);
    }
    if (id >= meshes.size()) meshes.resize(size_t(id) + 1);
    meshes[id] = std::move(m);
    vertices.allocate(id, meshes[id].vertexCount);
    indices.allocate(id, meshes[id].indexCount);
    order.push_back(id);
    meshNames[name] = id;
    registeredMeshes++;
    rebuildLayout();
}

void MeshPool::rebuildLayout()
{
    bucketList.clear();
    bucketBase.assign(meshes.size(), 0);
    for (int c = 0; c < pipelineClassCount; ++c)
    {
        ranges[c].first = uint32_t(bucketList.size());
        for (MeshId id : order)
        {
            const Mesh& m = meshes[id];
            if (pipelineClassIndex(materials[m.material].data) != c) continue;
            bucketBase[id] = uint32_t(bucketList.size());
            for (size_t l = 0; l < m.lods.size(); ++l) bucketList.push_back(Bucket{id, uint8_t(l), uint8_t(c)});
        }
        ranges[c].count = uint32_t(bucketList.size()) - ranges[c].first;
    }
    version++;
    recordsDirty = true;
}

void MeshPool::checkGrowth()
{
    // A growth moved the meshes: new first indices and vertex offsets in the records and the commands
    if (vertices.generation() != vertexGeneration || indices.generation() != indexGeneration)
    {
        vertexGeneration = vertices.generation();
        indexGeneration  = indices.generation();
        version++;
        recordsDirty = true;
    }
}

vk::DrawIndexedIndirectCommand MeshPool::command(uint32_t b) const
{
    const Bucket& bucket = bucketList[b];
    const Mesh& m        = meshes[bucket.mesh];
    const Lod& l         = m.lods[bucket.lod];
    vk::DrawIndexedIndirectCommand c;
    c.indexCount    = l.indexCount;
    c.instanceCount = 0;
    c.firstIndex    = uint32_t(indices.offset(bucket.mesh)) + l.firstIndex;
    c.vertexOffset  = int32_t(vertices.offset(bucket.mesh) + l.firstVertex);
    c.firstInstance = 0;
    return c;
}

bool MeshPool::recordGrowth(vk::CommandBuffer cmd)
{
    checkGrowth();
    bool copied = vertices.recordGrowth(cmd);
    copied      = indices.recordGrowth(cmd) || copied;
    return copied;
}

void MeshPool::queueUploads(UploadBatch& batch)
{
    checkGrowth();
    for (MeshId id : order)
    {
        Mesh& m = meshes[id];
        if (m.pendingVertices.empty()) continue;
        batch.add(vertices.buffer(), vertices.offset(id) * sizeof(MeshVertex), m.pendingVertices.data(),
                  m.pendingVertices.size() * sizeof(MeshVertex));
        batch.add(indices.buffer(), indices.offset(id) * sizeof(uint32_t), m.pendingIndices.data(),
                  m.pendingIndices.size() * sizeof(uint32_t));
        vertices.setResident(id);
        indices.setResident(id);
        m.pendingVertices = {};
        m.pendingIndices  = {};
    }
    if (recordsDirty || !meshRecords)
    {
        std::vector<GpuMeshRecord> records(std::max<size_t>(meshes.size(), 1));
        for (MeshId id : order)
        {
            const Mesh& m    = meshes[id];
            GpuMeshRecord& r = records[id];
            r.material       = m.material;
            r.lodCount       = uint32_t(m.lods.size());
            r.bucketBase     = bucketBase[id];
            r.pipelineClass  = pipelineClassIndex(materials[m.material].data);
            for (int i = 0; i < 3; ++i) r.sphere[i] = m.sphereCenter(i);
            r.sphere[3]  = m.sphereRadius;
            r.padding[0] = m.boundsPadding;
            for (size_t l = 0; l < m.lods.size(); ++l)
            {
                r.lods[l].firstIndex   = uint32_t(indices.offset(id)) + m.lods[l].firstIndex;
                r.lods[l].indexCount   = m.lods[l].indexCount;
                r.lods[l].vertexOffset = int32_t(vertices.offset(id) + m.lods[l].firstVertex);
                r.lods[l].endDistance  = m.lods[l].endDistance;
            }
        }
        const size_t bytes = records.size() * sizeof(GpuMeshRecord);
        ensureTable(*base, meshRecords, bytes);
        batch.add(meshRecords, 0, records.data(), bytes);
        recordsDirty = false;
    }
    if (materialsDirty || !materialTable)
    {
        std::vector<GpuMaterial> table(std::max<size_t>(materials.size(), 1));
        for (size_t i = 0; i < materials.size(); ++i)
            if (materials[i].registered) table[i] = toGpu(materials[i].data);
        const size_t bytes = table.size() * sizeof(GpuMaterial);
        ensureTable(*base, materialTable, bytes);
        batch.add(materialTable, 0, table.data(), bytes);
        materialsDirty = false;
    }
    if (paletteDirty || !paletteTable)
    {
        // x: sRGB8 RGB + linear alpha, y: roughness | metalness << 8; at least one (white) entry
        std::vector<uint32_t> table(std::max<size_t>(paletteEntries.size(), 1) * 2, 0);
        table[0] = 0xFFFFFFFFu;
        table[1] = 204;
        for (size_t i = 0; i < paletteEntries.size(); ++i)
        {
            table[2 * i]     = packColor(paletteEntries[i].color);
            table[2 * i + 1] = uint32_t(paletteEntries[i].roughness) | (uint32_t(paletteEntries[i].metalness) << 8);
        }
        // The shader clamps the index to the descriptor's range (paletteCount() entries)
        const size_t bytes = table.size() * sizeof(uint32_t);
        ensureTable(*base, paletteTable, bytes);
        batch.add(paletteTable, 0, table.data(), bytes);
        paletteDirty = false;
    }
}

}  // namespace Vulkan
}  // namespace Saiga
