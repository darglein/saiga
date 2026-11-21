/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// RecordingRenderSink (docs/engine_design.md §10.1, §17.2; S19b): a RenderSink without a device for tests and
// scenarios. It keeps state (the registered meshes and materials, the live dynamic instances with their last poses,
// counters), not a history, so long scenarios do not grow; an opt-in call log records every call for tests. It never
// changes the World (its hash).

#pragma once

#include "saiga/engine/sync/RenderSink.h"
#include "saiga/export.h"

#include <map>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class SAIGA_ENGINE_API RecordingRenderSink : public RenderSink
{
   public:
    struct Dynamic
    {
        MeshId mesh = 0;
        bool set    = false;  // setDynamic was called
        ChunkId chunk;
        DynamicPose prev, curr;
        uint64_t updates = 0;  // setDynamic calls
    };
    struct Counters
    {
        uint64_t registerMesh = 0, registerMaterial = 0, addDynamic = 0, setDynamic = 0, removeDynamic = 0;
        uint64_t addInstanceSet = 0, updateInstances = 0, removeInstance = 0, removeInstanceSet = 0;
        uint64_t setCamera = 0, setInterpolation = 0;
    };

    void registerMaterial(MaterialId id, const std::string& name, const MaterialData& data) override;
    void registerMesh(MeshId id, const std::string& name, std::shared_ptr<const MeshLods> lods,
                      MaterialId material) override;
    InstanceSetHandle addInstanceSet(const RegionId& region, const std::vector<StaticInstance>& instances,
                                     uint32_t capacity = 0) override;
    void updateInstances(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& v) override;
    void removeInstance(InstanceSetHandle set, uint32_t index) override;
    void removeInstanceSet(InstanceSetHandle set) override;
    InstanceHandle addDynamic(MeshId mesh) override;
    void setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev, const DynamicPose& curr,
                    uint32_t boneOffset = 0) override;
    void removeDynamic(InstanceHandle h) override;
    void setCamera(SpaceId space, const Vec3& position, const Quat& rotation, double fovy) override;
    void setInterpolation(float alpha) override;

    // ---- state ----
    // Live dynamic instances by handle ID
    const std::map<uint32_t, Dynamic>& dynamics() const { return live; }
    const Dynamic* dynamic(InstanceHandle h) const;
    size_t liveDynamics() const { return live.size(); }
    // Registered names by ID
    const std::map<MeshId, std::string>& meshes() const { return meshNames; }
    const std::map<MaterialId, std::string>& materials() const { return materialNames; }
    // The material of a registered mesh (0: none)
    MaterialId meshMaterial(MeshId id) const;
    const Counters& counters() const { return count; }
    size_t liveInstanceSets() const { return sets.size(); }
    float interpolation() const { return alpha; }

    // ---- opt-in call log (tests): one line per call, e.g. "addDynamic 256 -> 1", "setDynamic 1", "removeDynamic 1"
    void setLogging(bool on) { logging = on; }
    const std::vector<std::string>& log() const { return calls; }
    void clearLog() { calls.clear(); }

   private:
    // Callers check logging first (no message is formatted while the log is off)
    void note(const std::string& line) { calls.push_back(line); }
    std::map<uint32_t, Dynamic> live;
    std::map<MeshId, std::string> meshNames;
    std::map<MeshId, MaterialId> meshMaterials;
    std::map<MaterialId, std::string> materialNames;
    std::map<uint32_t, uint32_t> sets;  // set ID -> capacity
    uint32_t nextHandle = 1, nextSet = 1;
    Counters count;
    float alpha  = 0;
    bool logging = false;
    std::vector<std::string> calls;
};

}  // namespace Engine
}  // namespace Saiga
