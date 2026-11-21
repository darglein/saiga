/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// RecordingRenderSink: the RenderSink of tests and scenarios (state and counters, an opt-in call log).

#include "saiga/engine/sync/RecordingRenderSink.h"

#include "saiga/core/util/assert.h"

namespace Saiga
{
namespace Engine
{
void RecordingRenderSink::registerMaterial(MaterialId id, const std::string& name, const MaterialData&)
{
    auto it = materialNames.find(id);
    if (it != materialNames.end())
    {
        // The same sink set again: identical registrations are no-ops
        SAIGA_ASSERT(it->second == name, "RecordingRenderSink: material " + std::to_string(id) + " registered as \"" +
                                             it->second + "\" and \"" + name + "\"");
        return;
    }
    for (const auto& m : materialNames)
        SAIGA_ASSERT(m.second != name, "RecordingRenderSink: material \"" + name + "\" registered with two IDs");
    materialNames[id] = name;
    count.registerMaterial++;
    if (logging) note("registerMaterial " + std::to_string(id) + " " + name);
}

void RecordingRenderSink::registerMesh(MeshId id, const std::string& name, std::shared_ptr<const MeshLods> lods,
                                       MaterialId material)
{
    SAIGA_ASSERT(lods && !lods->lods.empty(), "RecordingRenderSink: mesh \"" + name + "\" without LODs");
    auto it = meshNames.find(id);
    if (it != meshNames.end())
    {
        SAIGA_ASSERT(it->second == name, "RecordingRenderSink: mesh " + std::to_string(id) + " registered as \"" +
                                             it->second + "\" and \"" + name + "\"");
        return;
    }
    for (const auto& m : meshNames)
        SAIGA_ASSERT(m.second != name, "RecordingRenderSink: mesh \"" + name + "\" registered with two IDs");
    SAIGA_ASSERT(materialNames.count(material), "RecordingRenderSink: mesh \"" + name + "\" of an unknown material");
    meshNames[id]     = name;
    meshMaterials[id] = material;
    count.registerMesh++;
    if (logging) note("registerMesh " + std::to_string(id) + " " + name);
}

MaterialId RecordingRenderSink::meshMaterial(MeshId id) const
{
    auto it = meshMaterials.find(id);
    return it == meshMaterials.end() ? MaterialId(0) : it->second;
}

InstanceSetHandle RecordingRenderSink::addInstanceSet(const RegionId&, const std::vector<StaticInstance>& instances,
                                                      uint32_t capacity)
{
    for (const StaticInstance& i : instances)
        SAIGA_ASSERT(meshNames.count(i.mesh), "RecordingRenderSink: an instance of an unregistered mesh");
    InstanceSetHandle h;
    h.id       = nextSet++;
    sets[h.id] = capacity ? capacity : uint32_t(instances.size());
    count.addInstanceSet++;
    if (logging) note("addInstanceSet -> " + std::to_string(h.id));
    return h;
}

void RecordingRenderSink::updateInstances(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& v)
{
    auto it = sets.find(set.id);
    SAIGA_ASSERT(it != sets.end() && first + v.size() <= it->second, "RecordingRenderSink: updateInstances");
    count.updateInstances++;
    if (logging) note("updateInstances " + std::to_string(set.id));
}

void RecordingRenderSink::removeInstance(InstanceSetHandle set, uint32_t index)
{
    auto it = sets.find(set.id);
    SAIGA_ASSERT(it != sets.end() && index < it->second, "RecordingRenderSink: removeInstance");
    count.removeInstance++;
    if (logging) note("removeInstance " + std::to_string(set.id));
}

void RecordingRenderSink::removeInstanceSet(InstanceSetHandle set)
{
    // Erased in every build (SAIGA_ASSERT does not evaluate its condition without SAIGA_ASSERTS)
    const size_t erased = sets.erase(set.id);
    SAIGA_ASSERT(erased == 1, "RecordingRenderSink: removeInstanceSet of an unknown set");
    count.removeInstanceSet++;
    if (logging) note("removeInstanceSet " + std::to_string(set.id));
}

InstanceHandle RecordingRenderSink::addDynamic(MeshId mesh)
{
    SAIGA_ASSERT(meshNames.count(mesh),
                 "RecordingRenderSink: addDynamic of the unregistered mesh " + std::to_string(mesh));
    InstanceHandle h;
    h.id       = nextHandle++;
    live[h.id] = Dynamic{mesh, false, ChunkId(), DynamicPose(), DynamicPose(), 0};
    count.addDynamic++;
    if (logging) note("addDynamic " + std::to_string(mesh) + " -> " + std::to_string(h.id));
    return h;
}

void RecordingRenderSink::setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev,
                                     const DynamicPose& curr, uint32_t)
{
    auto it = live.find(h.id);
    SAIGA_ASSERT(it != live.end(), "RecordingRenderSink: setDynamic of a removed handle");
    it->second.set   = true;
    it->second.chunk = chunk;
    it->second.prev  = prev;
    it->second.curr  = curr;
    it->second.updates++;
    count.setDynamic++;
    if (logging) note("setDynamic " + std::to_string(h.id));
}

void RecordingRenderSink::removeDynamic(InstanceHandle h)
{
    // Erased in every build (SAIGA_ASSERT does not evaluate its condition without SAIGA_ASSERTS)
    const size_t erased = live.erase(h.id);
    SAIGA_ASSERT(erased == 1, "RecordingRenderSink: removeDynamic of an unknown handle");
    count.removeDynamic++;
    if (logging) note("removeDynamic " + std::to_string(h.id));
}

const RecordingRenderSink::Dynamic* RecordingRenderSink::dynamic(InstanceHandle h) const
{
    auto it = live.find(h.id);
    return it == live.end() ? nullptr : &it->second;
}

void RecordingRenderSink::setCamera(SpaceId, const Vec3&, const Quat&, double)
{
    count.setCamera++;
}

void RecordingRenderSink::setInterpolation(float a)
{
    alpha = a;
    count.setInterpolation++;
}

}  // namespace Engine
}  // namespace Saiga
