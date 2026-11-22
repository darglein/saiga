/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/InstanceBuffer.h"

#include "saiga/vulkan/scene/MeshPool.h"
#include "saiga/vulkan/scene/UploadBatch.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Vulkan
{
namespace
{
// The offset slots fit into the low 24 bits of a compacted entry
constexpr uint32_t maxSlots = 1u << 24;

// Rotates v by the unit quaternion (as rotateQuat of instance_common.glsl)
Vec3 rotate(const quat& q, const Vec3& v)
{
    const Vec3 u(q.x(), q.y(), q.z());
    const Vec3 t = 2.0 * u.cross(v);
    return v + double(q.w()) * t + u.cross(t);
}
}  // namespace

void InstanceBuffer::init(VulkanBase& base, uint32_t staticCapacity, uint64_t maxBytes)
{
    destroy();
    buffer.init(base, sizeof(StaticInstance), staticCapacity, vk::BufferUsageFlagBits::eStorageBuffer, maxBytes,
                "static instances");
}

void InstanceBuffer::destroy()
{
    buffer.destroy();
    setMap.clear();
    dynamicList.clear();
    freeDynamics.clear();
    slotList.clear();
    freeSlots.clear();
    chunkSlots.clear();
}

uint32_t InstanceBuffer::allocateSlot(const Slot& s)
{
    uint32_t slot;
    if (!freeSlots.empty())
    {
        slot = freeSlots.back();
        freeSlots.pop_back();
    }
    else
    {
        slot = uint32_t(slotList.size());
        SAIGA_ASSERT(slot < maxSlots, "RenderScene: too many offset slots (instance sets and chunks of dynamics)");
        slotList.emplace_back();
    }
    slotList[slot] = s;
    return slot;
}

void InstanceBuffer::freeSlot(uint32_t slot)
{
    slotList[slot] = Slot();
    freeSlots.push_back(slot);
}

void InstanceBuffer::computeBounds(Set& s, const MeshPool& meshes)
{
    s.hasBounds = false;
    s.maxScale  = 0;
    s.meshes.clear();
    Vec3 lo = Vec3::Zero();
    Vec3 hi = Vec3::Zero();
    for (uint32_t i = 0; i < s.used; ++i)
    {
        const StaticInstance& inst = s.instances[i];
        SAIGA_ASSERT(inst.scale >= 0, "RenderScene: an instance scale must be >= 0 (0: a tombstone)");
        if (inst.scale == 0) continue;
        const MeshPool::Mesh* m = meshes.mesh(inst.mesh);
        SAIGA_ASSERT(m, "RenderScene: an instance of an unregistered mesh (register the mesh before the set)");
        const quat q        = inst.getRotation();
        const double scale  = double(inst.scale);
        const Vec3 center   = inst.getPosition().cast<double>() + rotate(q, m->sphereCenter.cast<double>() * scale);
        const double radius = double(m->sphereRadius) * scale;
        const Vec3 r(radius, radius, radius);
        if (!s.hasBounds)
        {
            lo = center - r;
            hi = center + r;
        }
        else
        {
            lo = lo.cwiseMin(center - r);
            hi = hi.cwiseMax(center + r);
        }
        s.hasBounds = true;
        s.maxScale  = std::max(s.maxScale, scale);
        if (s.meshes.empty() || s.meshes.back() != inst.mesh) s.meshes.push_back(inst.mesh);
    }
    std::sort(s.meshes.begin(), s.meshes.end());
    s.meshes.erase(std::unique(s.meshes.begin(), s.meshes.end()), s.meshes.end());
    s.spheresMin = lo;
    s.spheresMax = hi;
    padBounds(s, meshes);
}

void InstanceBuffer::padBounds(Set& s, const MeshPool& meshes)
{
    double padding   = 0;
    s.maxEndDistance = 0;
    for (MeshId id : s.meshes)
    {
        const MeshPool::Mesh* m = meshes.mesh(id);
        padding                 = std::max(padding, double(m->boundsPadding));
        s.maxEndDistance        = std::max(s.maxEndDistance, double(m->lastEndDistance()));
    }
    const double p = padding * s.maxScale;
    s.boundsMin    = s.spheresMin - Vec3(p, p, p);
    s.boundsMax    = s.spheresMax + Vec3(p, p, p);
}

void InstanceBuffer::refreshBounds(MeshId mesh, const MeshPool& meshes)
{
    for (auto& [id, s] : setMap)
        if (std::binary_search(s.meshes.begin(), s.meshes.end(), mesh)) padBounds(s, meshes);
}

InstanceSetHandle InstanceBuffer::addSet(const RegionId& region, const std::vector<StaticInstance>& instances,
                                         uint32_t capacity, const MeshPool& meshes)
{
    if (capacity == 0) capacity = uint32_t(instances.size());
    SAIGA_ASSERT(capacity >= instances.size(), "addInstanceSet: capacity < instances.size()");
    SAIGA_ASSERT(capacity > 0, "addInstanceSet: an empty set needs a capacity");
    const uint32_t id = nextSet++;
    Set s;
    s.region   = region;
    s.capacity = capacity;
    s.used     = uint32_t(instances.size());
    s.instances.resize(capacity);
    // The unused tail is zero (tombstones): never visited, the culling stops at used
    for (auto& i : s.instances) i.scale = 0;
    std::copy(instances.begin(), instances.end(), s.instances.begin());
    computeBounds(s, meshes);
    Slot slot;
    slot.kind   = Slot::Kind::Region;
    slot.region = region;
    s.slot      = allocateSlot(slot);
    if (s.used > 0) s.dirty.push_back({0, s.used});
    buffer.allocate(id, capacity);
    setMap[id] = std::move(s);
    return InstanceSetHandle{id};
}

void InstanceBuffer::update(InstanceSetHandle h, uint32_t first, const std::vector<StaticInstance>& instances,
                            const MeshPool& meshes)
{
    auto it = setMap.find(h.id);
    SAIGA_ASSERT(it != setMap.end(), "updateInstances: unknown instance set");
    Set& s = it->second;
    SAIGA_ASSERT(uint64_t(first) + instances.size() <= s.capacity, "updateInstances: first + size > capacity");
    if (instances.empty()) return;
    std::copy(instances.begin(), instances.end(), s.instances.begin() + first);
    // A gap between the used size and first becomes visited: its tombstones are uploaded too
    if (first > s.used) s.dirty.push_back({s.used, first - s.used});
    s.used = std::max(s.used, first + uint32_t(instances.size()));
    s.dirty.push_back({first, uint32_t(instances.size())});
    computeBounds(s, meshes);
}

void InstanceBuffer::removeInstance(InstanceSetHandle h, uint32_t index)
{
    auto it = setMap.find(h.id);
    SAIGA_ASSERT(it != setMap.end(), "removeInstance: unknown instance set");
    Set& s = it->second;
    SAIGA_ASSERT(index < s.capacity, "removeInstance: index out of range");
    s.instances[index].scale = 0;
    s.dirty.push_back({index, 1});
}

void InstanceBuffer::removeSet(InstanceSetHandle h)
{
    auto it = setMap.find(h.id);
    SAIGA_ASSERT(it != setMap.end(), "removeInstanceSet: unknown instance set");
    freeSlot(it->second.slot);
    buffer.free(h.id);
    setMap.erase(it);
}

uint32_t InstanceBuffer::acquireChunk(const ChunkId& chunk)
{
    auto it = chunkSlots.find(chunk);
    if (it != chunkSlots.end())
    {
        it->second.second++;
        return it->second.first;
    }
    // The map entry first, undone if the slot allocation throws (RenderSink's failure contract)
    it = chunkSlots.emplace(chunk, std::pair<uint32_t, uint32_t>(0, 1)).first;
    Slot s;
    s.kind  = Slot::Kind::Chunk;
    s.chunk = chunk;
    try
    {
        it->second.first = allocateSlot(s);
    }
    catch (...)
    {
        chunkSlots.erase(it);
        throw;
    }
    return it->second.first;
}

void InstanceBuffer::releaseChunk(const ChunkId& chunk)
{
    auto it = chunkSlots.find(chunk);
    SAIGA_ASSERT(it != chunkSlots.end());
    if (--it->second.second == 0)
    {
        freeSlot(it->second.first);
        chunkSlots.erase(it);
    }
}

InstanceHandle InstanceBuffer::addDynamic(MeshId mesh)
{
    uint32_t index;
    if (!freeDynamics.empty())
    {
        index = freeDynamics.back();
        freeDynamics.pop_back();
    }
    else
    {
        index = uint32_t(dynamicList.size());
        SAIGA_ASSERT(index < 0x7FFFFFFFu, "RenderScene: too many dynamic instances");
        dynamicList.emplace_back();
    }
    Dynamic d;
    d.alive            = true;
    d.mesh             = mesh;
    dynamicList[index] = d;
    return InstanceHandle{index + 1};
}

void InstanceBuffer::setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev,
                                const DynamicPose& curr, uint32_t boneOffset)
{
    SAIGA_ASSERT(h.id > 0 && h.id <= dynamicList.size() && dynamicList[h.id - 1].alive,
                 "setDynamic: unknown dynamic instance");
    Dynamic& d = dynamicList[h.id - 1];
    if (!d.set || d.chunk != chunk)
    {
        // Room for the old chunk's slot first: after acquireChunk nothing may throw (RenderSink's failure contract)
        if (d.set && freeSlots.size() == freeSlots.capacity())
            freeSlots.reserve(std::max<size_t>(16, 2 * freeSlots.capacity()));
        const uint32_t slot = acquireChunk(chunk);
        if (d.set) releaseChunk(d.chunk);
        d.slot  = slot;
        d.chunk = chunk;
    }
    d.set        = true;
    d.prev       = prev;
    d.curr       = curr;
    d.boneOffset = boneOffset;
}

void InstanceBuffer::removeDynamic(InstanceHandle h)
{
    SAIGA_ASSERT(h.id > 0 && h.id <= dynamicList.size() && dynamicList[h.id - 1].alive,
                 "removeDynamic: unknown dynamic instance");
    // Everything that can allocate first: a removal that throws changes nothing (RenderSink's failure contract)
    auto room = [](std::vector<uint32_t>& v)
    {
        if (v.size() == v.capacity()) v.reserve(std::max<size_t>(16, 2 * v.capacity()));
    };
    room(freeDynamics);
    room(freeSlots);
    Dynamic& d = dynamicList[h.id - 1];
    if (d.set) releaseChunk(d.chunk);
    d = Dynamic();
    freeDynamics.push_back(h.id - 1);
}

bool InstanceBuffer::moving(const Dynamic& d)
{
    return d.prev.position != d.curr.position || d.prev.scale != d.curr.scale ||
           d.prev.rotation.coeffs() != d.curr.rotation.coeffs();
}

void InstanceBuffer::writeDynamics(GpuDynamic* out) const
{
    for (size_t i = 0; i < dynamicList.size(); ++i)
    {
        const Dynamic& d = dynamicList[i];
        GpuDynamic g;
        if (d.alive && d.set)
        {
            auto pose = [](const DynamicPose& p, float* pos, float* rot)
            {
                for (int k = 0; k < 3; ++k) pos[k] = float(p.position(k));
                pos[3]       = float(p.scale);
                const Quat q = p.rotation.normalized();
                rot[0]       = float(q.x());
                rot[1]       = float(q.y());
                rot[2]       = float(q.z());
                rot[3]       = float(q.w());
            };
            pose(d.prev, g.prevPos, g.prevRot);
            pose(d.curr, g.currPos, g.currRot);
            g.mesh       = d.mesh;
            g.slot       = d.slot;
            g.boneOffset = d.boneOffset;
            // S36b: bit 0 the instance moves (its pair differs): the atlas's dynamic layer, not its cached static one
            g.pad = moving(d) ? 1u : 0u;
        }
        out[i] = g;
    }
}

void InstanceBuffer::queueUploads(UploadBatch& batch)
{
    for (auto& [id, s] : setMap)
    {
        if (s.dirty.empty()) continue;
        // Merged into disjoint ranges (one copy may not write a range twice), uploaded from the CPU copy
        std::sort(s.dirty.begin(), s.dirty.end());
        std::vector<std::pair<uint32_t, uint32_t>> merged;  // [begin, end)
        for (auto [first, count] : s.dirty)
        {
            if (!merged.empty() && first <= merged.back().second)
                merged.back().second = std::max(merged.back().second, first + count);
            else
                merged.push_back({first, first + count});
        }
        const uint64_t base = buffer.offset(id);
        for (auto [begin, end] : merged)
        {
            batch.add(buffer.buffer(), (base + begin) * sizeof(StaticInstance), s.instances.data() + begin,
                      size_t(end - begin) * sizeof(StaticInstance));
        }
        s.dirty.clear();
        buffer.setResident(id);
    }
}

}  // namespace Vulkan
}  // namespace Saiga
