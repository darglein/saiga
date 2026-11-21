/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainSystem (docs/engine_design.md §12.2, §12.3; S20), owned by the World (World::terrain()): the terrain of the
// World's spaces. The exact surface height (World::terrainHeight) from base-grid samples (a resident level-0 tile or
// the space's height source), the visual streaming of the view camera's space on the streaming pool (tiles, the
// pyramid, grass masks in a TerrainStore per space), the single send point of everything that reaches the render sink,
// the GPU terrain set's rebuild, readiness and the complete-view loop of the app's complete points.

#pragma once

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/terrain/SampleRect.h"
#include "saiga/engine/terrain/TerrainDirtyQueue.h"
#include "saiga/engine/world/SpaceTemplate.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace Saiga
{
namespace Engine
{
class RenderSink;
class StreamingPool;
class TerrainStore;
class TerrainEditStore;
class TerrainVisualPatch;

// The visual streaming policy (R-MECH-1): EngineAppSetup / game code may change it (TerrainSystem::setViewSettings)
struct TerrainViewSettings
{
    // Level k tiles are desired within residencyRange0 * 2^k of TerrainPass's LOD distance: sqrt(dxz^2 + hc^2), dxz the
    // horizontal distance from the camera to the tile, hc the camera height above the surface under it
    double residencyRange0 = 400;
    // The budget per streamed level = GPU atlas layers (part of the layout)
    int tilesPerLevel = 48;
    // The budget of grass masks = GPU mask layers (part of the layout)
    int grassMaskLayers = 16;
    // Tile sink calls per updateView (completeView sends everything)
    int uploadsPerFrame = 8;
    // Chunks within this radius get grass masks
    double grassRadius     = 80;
    double grassMinNormalY = 0.82, grassMinAboveSea = 0.4;
    int workers = 2, maxInFlight = 8;
    // Terrain edits (S22): the samples patched per updateView (summed over the levels and the pyramid; about 1 MB of
    // uploads), the edited grass masks submitted per updateView (completeView: no limit)
    int64_t editSamplesPerFrame = int64_t(1) << 18;
    int editMasksPerFrame       = 2;
};

// One space of the world (SpaceId = index; S23a: the generation pipeline's instances): its template, its seed, its
// height source (the pipeline's composed height) and whether it has terrain (World::spaceHasTerrain)
struct TerrainSpace
{
    SpaceTemplate settings;
    uint64_t seed = 0;
    TerrainHeightSource source;
    bool hasTerrain = false;
};

class SAIGA_ENGINE_API TerrainSystem
{
   public:
    explicit TerrainSystem(std::vector<TerrainSpace> spaces);
    // Destroys the pool first: no job outlives the data its completion would touch
    ~TerrainSystem();
    TerrainSystem(const TerrainSystem&)            = delete;
    TerrainSystem& operator=(const TerrainSystem&) = delete;

    // World::terrainHeight (§7.10, §12.2): x and z clamped to the space [0, sizeChunks * 256]; the cell of the base
    // grid (the last cell at the far edge), the three samples of its triangle (editedSample: base + terrain edit,
    // S22) and terrainSurfaceDouble, in metres. 0 for a space without terrain while it has no edits (D1). Const and
    // thread safe (parallelEach)
    double height(SpaceId space, double x, double z) const;
    // A base-grid sample of the BASE data: the resident level-0 tile or the space's height source; 0 for a space
    // without terrain. A cheap point query, const and thread safe (parallelEach). S23a replaces only the source
    int32_t sample(SpaceId space, int64_t sx, int64_t sz) const;
    // World::terrainSample (S22): clampTerrainHeight(sample + the edit delta); the delta 0 without edits. Never
    // allocates; const and thread safe while the edit store is not mutated
    int32_t editedSample(SpaceId space, int64_t sx, int64_t sz) const;
    // The base samples read from the height source instead of a resident level-0 tile, by every caller of sample
    // (a relaxed counter; S20's pool generation does not go through sample and is not counted). Tests and the
    // terraform benchmark read differences of it to prove that a path calls no hook
    uint64_t baseHookSamples() const { return hookSamples.load(std::memory_order_relaxed); }
    // Engine-internal: the World's terrain edits (read by height and the visual merge; S22)
    void setEdits(const TerrainEditStore* store) { edits = store; }
    // Engine-internal (World::onTerrainChanged, S22): a changed box of global base samples. Only while a sink is
    // attached: a tile box if the space is the view space, a pyramid box of the space (headless worlds keep nothing)
    void terrainChanged(SpaceId space, const SampleRect& box);
    // The visual dirty queue (tests)
    const TerrainDirtyQueue& dirtyQueue() const { return dirty; }
    // The edited grass masks (tests)
    const TerrainVisualPatch& visualPatch() const { return *visual; }

    const SpaceTemplate& spaceTemplate(SpaceId space) const;
    // E6: the space has terrain (its generator's hasTerrain; S23a)
    bool hasTerrain(SpaceId space) const;
    const TerrainHeightSource& heightSource(SpaceId space) const;
    // The seed of the construction (World::spaceSeed follows the plans of a load)
    uint64_t spaceSeed(SpaceId space) const;
    size_t spaceCount() const { return spaces.size(); }

    // Main thread, between steps (EngineApp, every rendered frame, with the view camera). In order:
    //  1. drains the pool: finished tiles / masks / pyramids into the store (an exception is fatal);
    //  2. a space other than the current view space (or a changed layout) starts a new GPU terrain set; a view space
    //     without terrain sends clearTerrain() once (if anything was sent) and stops here;
    //  3. the desired set: while the space's pyramid is not in the store only the pyramid is requested (readiness 0);
    //     otherwise per streamed level the tiles within the range, nearest first, truncated to tilesPerLevel; grass
    //     masks for the chunks (of a desired level-0 tile) whose box (pyramid heights +- 64 m) is within grassRadius
    //     (3D: the grass fades by its 3D distance), truncated to grassMaskLayers;
    //  4. requests: missing desired tiles by distance / range ratio (coarse levels first at equal ratio), then the
    //     masks whose level-0 tile is in the store, at most maxInFlight in flight; requests no longer desired are
    //     cancelled;
    //  5. evicts tiles that are not desired, LRU first, so that each level holds at most tilesPerLevel (and the masks
    //     that are not desired);
    //  6. sends every removal of 5, then at most uploadsPerFrame tile additions by distance and the mask additions.
    // Without a render sink, or for a non-finite camera, nothing is requested or sent
    void updateView(SpaceId space, const Vec3& camera);
    // Blocks until the desired set of this camera is resident and sent (the pyramid, then the tiles, then the masks,
    // repeated until nothing is missing; no maxInFlight or uploadsPerFrame limit; the calling thread runs queued jobs
    // too). Returns false on timeout (what is resident is still sent; no caller job starts after the deadline, a
    // running one is finished), at once when an iteration brought nothing new, submitted no job and nothing is queued
    // or running (no later one could: logged with what is missing), or for a non-finite camera. Without a sink it fills
    // the store only. A space without terrain returns true at once
    bool completeView(SpaceId space, const Vec3& camera, double timeoutSeconds = 120);
    // Nothing of this camera's desired set is missing (resident and sent): completeView would generate and send
    // nothing of the desired set. True for a space without terrain; without a sink only the residency counts. Queued
    // patches of terrain edits (the dirty queue) do not count: they wait for no generation, and updateView sends them
    // within its budget (the benchmark mode's GPU wait before a streaming frame keys on this)
    bool viewComplete(SpaceId space, const Vec3& camera) const;
    // D6: the fraction of (the pyramid + the desired tiles of this camera within radius, the residency's distance) that
    // have been sent to the sink; 0 while the pyramid is not sent (or without a sink); 1 for a space without terrain
    double readiness(SpaceId space, const Vec3& camera, double radius) const;
    // The sink's bytes not yet recorded (RenderSink::terrainUploadsPending); 0 without a sink
    size_t terrainUploadsPending() const;
    // tilesPerLevel / grassMaskLayers change the layout (a new GPU set at the next update; the resident tiles are
    // re-sent, truncated to the new budget); the grass thresholds drop and regenerate every mask; the rest applies from
    // the next update (workers when the pool is created)
    void setViewSettings(const TerrainViewSettings& settings);
    const TerrainViewSettings& viewSettings() const { return view; }
    // The template + the current settings
    Vulkan::TerrainLayout layout(SpaceId space) const;
    // The CPU cache; null before the space's first view; the same pointer while the world lives (D5)
    const Vulkan::TerrainPyramidData* pyramid(SpaceId space) const;

    struct LevelStats
    {
        int desired = 0, truncated = 0, resident = 0, sent = 0;
    };
    struct Stats
    {
        std::array<LevelStats, 8> levels{};
        int masksDesired = 0, masksTruncated = 0, masksResident = 0, masksSent = 0;
        // Requests in flight, tile additions sent by the last update
        int pending = 0, uploadsLastUpdate = 0;
        // Every finished generation (tiles, masks, pyramids); the pyramids alone; evicted tiles
        uint64_t generated = 0, pyramidsGenerated = 0, evicted = 0;
        // The stores (tiles, pyramids, masks) and the results in flight
        size_t cpuBytes   = 0;
        SpaceId viewSpace = 0;
        // The time of the last completeView
        double lastCompleteMs = 0;
        // Terrain edits (S22): patched samples and patch calls (cumulative and of the last update), the queued dirty
        // boxes, the sends that were merged copies, the edited grass masks computed and the results dropped (the
        // chunk's terrain changed meanwhile); the time of the visual part of the last update / completeView (the
        // merge at the send point, the patch step, the mask completions: the frame profile's terrain/visual)
        uint64_t patchedSamples = 0, patchCalls = 0, mergedSends = 0, maskRecomputes = 0, maskResultsDropped = 0;
        uint64_t patchedSamplesLastUpdate = 0;
        // Edited masks submitted to the pool by the last update, queued recomputes, in flight
        int maskJobsLastUpdate = 0, maskQueue = 0, maskJobsInFlight = 0;
        size_t queuedBoxes  = 0;
        double lastVisualMs = 0;
    };
    const Stats& stats() const { return statistics; }
    // The lazy StreamingPool exists (a sink alone starts none)
    bool poolCreated() const { return pool != nullptr; }

    // Engine-internal (a host load with other generator pins, S23a): new height sources per space; everything derived
    // from the old ones is dropped (the pool's jobs, the stores incl. the pyramids, the edited masks, the dirty queue)
    // and the sink's terrain is cleared (clearTerrain): the view's pyramid and tiles are requested again
    void replaceSources(std::vector<TerrainHeightSource> sources);
    // Engine-internal (S23b): the content part of updateView / completeView (the World's ContentRenderSystem; it runs
    // whether or not the view space has terrain): called with the view space, the camera and whether the view is
    // completed
    void setContentView(std::function<void(SpaceId, const Vec3&, bool)> fn) { contentView = std::move(fn); }
    // Engine-internal (S23b): the streaming pool (created on demand) for the content sets' jobs
    StreamingPool& streamingPool()
    {
        ensurePool();
        return *pool;
    }
    // Engine-internal: the World's render sink (World::setRenderSink). A new sink receives the pyramid and the tiles
    // and masks of the last view camera's desired set (current budget) that the store of the view space holds (through
    // the send point); a removed sink drops the sent state. A sink attached again first gets clearTerrain if it kept a
    // set when it was removed
    void setSink(RenderSink* sink);
    // Tests: a space's store
    const TerrainStore& store(SpaceId space) const;
    // Tests: while held, no tile or mask request is issued (a desired set that cannot progress: completeView's
    // no-progress return)
    void holdRequestsForTests(bool hold) { requestsHeld = hold; }

   private:
    struct Space
    {
        SpaceTemplate settings;
        uint64_t seed = 0;
        TerrainHeightSource source;
        bool hasTerrain = false;
        std::unique_ptr<TerrainStore> store;
    };
    // A request: kind (0 pyramid, 1 tile, 2 mask), space, level, x, z
    using Key = std::tuple<int, SpaceId, int, int32_t, int32_t>;
    struct Desired
    {
        // Per level, nearest first (truncated), with the distance
        std::array<std::vector<std::pair<double, RegionId>>, 8> tiles;
        std::array<int, 8> truncated{};
        std::vector<std::pair<double, ChunkId>> masks;
        int masksTruncated = 0;
        std::set<RegionId> tileSet;
        std::set<ChunkId> maskSet;
    };

    std::vector<Space> spaces;
    const TerrainEditStore* edits = nullptr;
    std::function<void(SpaceId, const Vec3&, bool)> contentView;
    // The terrain part of updateView / completeView
    void updateTerrainView(SpaceId space, const Vec3& camera);
    bool completeTerrainView(SpaceId space, const Vec3& camera, double timeoutSeconds);
    TerrainDirtyQueue dirty;
    // The visual time of the running update (S22)
    double visualMs = 0;
    // S22: the edited grass masks: the cache, the queued recomputes (insertion order, unique), the jobs in flight, the
    // GPU set the jobs were submitted for (incremented with every new set) and the current desired mask set
    std::unique_ptr<TerrainVisualPatch> visual;
    std::vector<ChunkId> maskQueue;
    std::set<ChunkId> editedJobs;
    uint64_t setGeneration = 0;
    std::set<ChunkId> desiredMasks;
    // Sent masks of chunks whose deltas returned to zero: the store's base mask replaces the edited one once resident
    std::set<ChunkId> baseResend;
    // A chunk's mask is current: the cached edited one (a chunk with deltas) or the store's base one
    const Vulkan::GrassMaskData* readyMask(const ChunkId& chunk) const;
    bool editedChunk(const ChunkId& chunk) const;
    void queueMask(const ChunkId& chunk);
    // Submits at most limit queued recomputes (< 0: all); returns the submissions
    int processMaskQueue(int limit);
    void requestEditedMask(const ChunkId& chunk);
    // S22: the patch step of the view space (budget in samples; < 0: everything); returns the patched samples
    int64_t patchEditedTiles(SpaceId space, int64_t budget);
    // The samples a box patches in the sent tiles / the pyramid (patch: send them)
    int64_t patchTileBox(SpaceId space, const SampleRect& box, bool patch);
    int64_t patchPyramidBox(SpaceId space, const SampleRect& box, bool patch);
    std::vector<int32_t> patchScratch;
    mutable std::atomic<uint64_t> hookSamples{0};
    TerrainViewSettings view;
    RenderSink* sink = nullptr;
    std::unique_ptr<StreamingPool> pool;
    std::map<Key, uint64_t> requests;
    // The jobs submitted so far (completeView's progress check)
    uint64_t jobsSubmitted = 0;
    bool requestsHeld      = false;
    // The sink's set: active once a pyramid was sent
    bool setActive   = false;
    SpaceId setSpace = 0;
    Vulkan::TerrainLayout setLayout;
    std::set<RegionId> sentTiles;
    std::set<ChunkId> sentMasks;
    // Sinks that were detached (or set again) while holding a set: no removal calls reached them, so the set is dropped
    // when one is attached again (compared by address only, never dereferenced while detached)
    std::vector<RenderSink*> staleSinks;
    bool viewSpaceValid = false;
    SpaceId viewSpace   = 0;
    // The camera of the last view update (setSink sends its desired set)
    Vec3 viewCamera = Vec3::Zero();
    uint64_t frame  = 0;
    // Incremented when the grass thresholds change: older mask results are dropped
    uint64_t maskGeneration = 0;
    Stats statistics;

    void ensurePool();
    void drainPool();
    // Step 2: the GPU terrain set of the view space
    void beginView(SpaceId space);
    Desired computeDesired(SpaceId space, const Vec3& camera) const;
    void markDesired(SpaceId space, const Desired& d);
    void cancelUndesired(SpaceId space, const Desired& d);
    void requestPyramid(SpaceId space);
    void requestTile(const RegionId& region, int priority);
    void requestMask(const ChunkId& chunk, int priority);
    // Missing desired tiles (at most limit requests in flight), then masks
    void issueRequests(SpaceId space, const Desired& d, size_t limit);
    void evict(SpaceId space, const Desired& d);
    // Removals, then at most limit tile additions by distance, then the mask additions
    void sendUpdates(SpaceId space, const Desired& d, int limit);
    bool missing(SpaceId space, const Desired& d) const;
    // The desired tiles and masks not in the store, or stored and not sent to the sink
    struct MissingCounts
    {
        int tilesNotStored = 0, tilesNotSent = 0, masksNotStored = 0, masksNotSent = 0;
        int total() const { return tilesNotStored + tilesNotSent + masksNotStored + masksNotSent; }
    };
    MissingCounts missingCounts(SpaceId space, const Desired& d) const;
    // Runs queued jobs on this thread until the deadline, then waits for the workers' jobs until it
    void runUntil(double deadlineSeconds);
    void updateStats(SpaceId space, const Desired* d);
    // The single send point (D3, E3): S22 merges its edits here
    void sendPyramid(SpaceId space, const Vulkan::TerrainPyramidData& pyramid);
    void sendTile(const RegionId& region, const Vulkan::TerrainTileData& tile);
    void sendGrassMask(const ChunkId& chunk, const Vulkan::GrassMaskData& mask);
    std::string describe(const Key& key) const;
};

}  // namespace Engine
}  // namespace Saiga
