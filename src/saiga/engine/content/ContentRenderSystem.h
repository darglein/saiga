/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ContentRenderSystem (docs/engine_design.md §10.1, §12.1; S23b), owned by the World (World::contentRender()): the
// content render sets. One static instance set per chunk (RegionId{space, 0, x, z}) with the chunk's tier-1 instances
// in (slot, counter) order, then the static parts of the plan's sites in the chunk (key slot 255, counter = the part's
// index), tombstoned keys at scale 0; a chunk with neither gets no set. Only spaces whose generator has slots or whose
// plan has static parts get sets.
//
// Who drives it (ContentRenderSettings::centres): None (the default: nothing happens, a sink alone builds nothing),
// View (EngineApp's rendered world: TerrainSystem::updateView / completeView call viewUpdate with the camera; the
// chunks of the camera's space within fullRadius, generated on the streaming pool and budgeted; completeView requests
// every missing one and runs the pool with the calling thread until they are built) or Players (ScenarioWorld: a pass
// in phase 8 after the render extract, the chunks within fullRadius of every Player of this world, generated
// synchronously). A set is removed when its chunk is more than fullRadius + 256 m from every centre. Record changes
// (World::onRecordChanged) reach the sink at once in both modes: a placement version change removes the set and
// requests it again, a tombstone change is diffed (removeInstance, or updateInstances with the original instance so it
// reappears). Pool results carry the request counter and version they started with and are dropped when either changed;
// tombstones and re-snaps never drop a result (the drain applies the current state). Terrain edits re-snap ReSnapOnEdit
// instances (step 7: the changed boxes of World::onTerrainChanged, queued per chunk, drained in viewUpdate / the pass).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/Types.h"
#include "saiga/engine/content/ChunkContent.h"
#include "saiga/engine/terrain/SampleRect.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;
class RenderSink;
class StreamingPool;

enum class ContentCentres : uint8_t
{
    None,
    View,
    Players
};

struct ContentRenderSettings
{
    ContentCentres centres = ContentCentres::None;
    // View: new set requests submitted per viewUpdate (the pool results are applied as they arrive)
    int setsPerFrame = 4;
};

class SAIGA_ENGINE_API ContentRenderSystem
{
   public:
    explicit ContentRenderSystem(World& world);
    ~ContentRenderSystem();
    ContentRenderSystem(const ContentRenderSystem&)            = delete;
    ContentRenderSystem& operator=(const ContentRenderSystem&) = delete;

    // A mode other than None needs a render sink (asserted); None removes every set
    void setSettings(const ContentRenderSettings& settings);
    const ContentRenderSettings& settings() const { return config; }
    // World::setRenderSink: the old sink's sets are removed, the next update builds them on the new one (no sink: the
    // centres become None)
    void setSink(RenderSink* sink);
    // A host load with other generator pins (S23a's replacePlans): every set is removed and built again
    void dropAll();

    // Players: the phase-8 pass (World::step after the render extract; not in stepNetworkOnly)
    void playersPass();
    // View (TerrainSystem::updateView / completeView): complete builds every desired set before it returns (the sets
    // are generated on the streaming pool, its workers and the calling thread), none is requested after the deadline
    // (steady_clock seconds, completeView's); false if a desired set of a completed view is not built
    bool viewUpdate(SpaceId space, const Vec3& camera, bool complete,
                    double deadline = std::numeric_limits<double>::infinity());
    // World::onRecordChanged: immediate in both modes (also inside stepNetworkOnly)
    void recordChanged(const ChunkId& chunk);
    // World::onTerrainChanged (step 7): the re-snap queue, only while sets exist
    void terrainChanged(SpaceId space, const SampleRect& box);
    // E8: the fraction of the desired sets of the space within radius of camera that are built (an empty chunk counts
    // as built); 1 when nothing is desired or the mode is not View
    double readiness(SpaceId space, const Vec3& camera, double radius) const;

    // Whether a space gets sets at all (its generator has slots or its plan static parts)
    bool spaceHasContent(SpaceId space) const;
    // The instances of a chunk's set as built now from its placement at version (tests, the golden): tier 1, then the
    // static parts; tombstones and re-snaps not applied
    std::vector<Vulkan::StaticInstance> buildInstances(const ChunkId& chunk, uint16_t version,
                                                       std::vector<InstanceKey>* keys = nullptr);
    // The live set of a chunk (invalid: none)
    Vulkan::InstanceSetHandle setOf(const ChunkId& chunk) const;
    size_t liveSets() const;

    struct Stats
    {
        uint64_t passes = 0, poolJobs = 0, setsBuilt = 0, setsRemoved = 0, emptyChunks = 0, resultsDropped = 0;
        uint64_t tombstonesHidden = 0, instancesRestored = 0, regenerated = 0, resnapped = 0, unknownMeshes = 0;
    };
    const Stats& stats() const { return statistics; }
    // Tests: runs in every pool job before it generates (e.g. waits on a latch)
    std::function<void()> jobHookForTests;

   private:
    struct ChunkSet
    {
        Vulkan::InstanceSetHandle handle;
        bool live = false, empty = false, requested = false;
        uint64_t counter = 0;  // incremented when the set is requested or removed
        uint16_t version = 0;  // of the live set or of the request
        uint64_t job     = 0;  // the pool job of the request (0: none)
        std::vector<Vulkan::StaticInstance> original;
        std::vector<Vulkan::StaticInstance> current;  // what the sink holds
        std::vector<InstanceKey> keys;                // per index (sorted)
        std::vector<InstanceKey> applied;             // the tombstones applied to the live set (sorted)
        std::vector<int32_t> localX, localZ;          // per tier-1 index, 1/256 m (re-snap)
        std::vector<uint8_t> resnap;                  // per index: a ReSnapOnEdit instance
        std::vector<SampleRect> dirty;                // queued re-snap boxes (global samples)
    };

    World& w;
    ContentRenderSettings config;
    RenderSink* sink = nullptr;
    std::map<ChunkId, ChunkSet> sets;
    std::set<std::tuple<std::string, uint32_t, std::string>> loggedMeshes;
    bool haveViewSpace = false;
    SpaceId viewSpace  = 0;
    Stats statistics;
    // The last request counter of this system (ChunkSet::counter; main thread)
    uint64_t requestCounter = 0;
    uint64_t nextCounter() { return ++requestCounter; }

    // The desired chunks around one centre (squares intersecting the disc of radius r)
    void chunksAround(SpaceId space, const Vec3& centre, double r, std::set<ChunkId>& out) const;
    double fullRadius(SpaceId space) const;
    // Removes the sets whose chunk is farther than fullRadius + 256 m from every centre
    void removeFar(const std::vector<std::pair<SpaceId, Vec3>>& centres);
    void removeSet(const ChunkId& chunk, ChunkSet& s);
    void removeAll();
    // buildInstances over a given placement (null: no tier-1 instances)
    std::vector<Vulkan::StaticInstance> buildFromPlacement(const ChunkId& chunk, const ChunkPlacement* placement,
                                                           std::vector<InstanceKey>* keys);
    // Builds a set from a placement: tombstones and re-snaps of the current state applied, then addInstanceSet
    void install(const ChunkId& chunk, ChunkSet& s, const ChunkPlacement& placement);
    void buildNow(const ChunkId& chunk, ChunkSet& s);
    void request(const ChunkId& chunk, ChunkSet& s, StreamingPool& pool);
    // viewUpdate with complete: requests every missing set of order (distance, chunk) and runs the pool until they are
    // installed or the deadline passed
    bool completeSets(const std::vector<std::pair<double, ChunkId>>& order, double deadline);
    // The original instance i with the current surface (ReSnapOnEdit) and the tombstone (scale 0)
    Vulkan::StaticInstance instanceNow(const ChunkId& chunk, const ChunkSet& s, size_t i, bool tombstoned) const;
    void drainResnaps();
    Vulkan::MeshId meshOf(const std::string& generator, uint32_t slot, const std::string& mesh);
};

}  // namespace Engine
}  // namespace Saiga
