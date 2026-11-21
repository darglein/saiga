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
#include "saiga/engine/world/SpaceTemplate.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <array>
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

// The visual streaming policy (R-MECH-1): EngineAppSetup / game code may change it (TerrainSystem::setViewSettings)
struct TerrainViewSettings
{
    // Level k tiles are desired within residencyRange0 * 2^k (3D distance from the camera to the tile box; heights from
    // the pyramid + 64 m margin)
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
};

class SAIGA_ENGINE_API TerrainSystem
{
   public:
    // One entry per space (SpaceId = index): the template and the space seed
    TerrainSystem(const std::vector<SpaceTemplate>& templates, const std::vector<uint64_t>& seeds);
    // Destroys the pool first: no job outlives the data its completion would touch
    ~TerrainSystem();
    TerrainSystem(const TerrainSystem&)            = delete;
    TerrainSystem& operator=(const TerrainSystem&) = delete;

    // World::terrainHeight (§7.10, §12.2): x and z clamped to the space [0, sizeChunks * 256]; the cell of the base
    // grid (the last cell at the far edge), the three samples of its triangle (sample) and terrainSurfaceDouble, in
    // metres. 0 for a space without terrain. Const and thread safe (parallelEach)
    double height(SpaceId space, double x, double z) const;
    // A base-grid sample of the BASE data: the resident level-0 tile or the space's height source; 0 for a space
    // without terrain. A cheap point query, const and thread safe (parallelEach). S22's World::terrainSample = this +
    // the edit delta; S23a replaces only the source
    int32_t sample(SpaceId space, int64_t sx, int64_t sz) const;

    const SpaceTemplate& spaceTemplate(SpaceId space) const;
    const TerrainHeightSource& heightSource(SpaceId space) const;
    uint64_t spaceSeed(SpaceId space) const;
    size_t spaceCount() const { return spaces.size(); }

    // Main thread, between steps (EngineApp, every rendered frame, with the view camera). In order:
    //  1. drains the pool: finished tiles / masks / pyramids into the store (an exception is fatal);
    //  2. a space other than the current view space (or a changed layout) starts a new GPU terrain set; a view space
    //     without terrain sends clearTerrain() once (if anything was sent) and stops here;
    //  3. the desired set: while the space's pyramid is not in the store only the pyramid is requested (readiness 0);
    //     otherwise per streamed level the tiles within the range, nearest first, truncated to tilesPerLevel; grass
    //     masks for the chunks within grassRadius (of a desired level-0 tile), truncated to grassMaskLayers;
    //  4. requests: missing desired tiles by distance / range ratio (coarse levels first at equal ratio), then the
    //     masks whose level-0 tile is in the store, at most maxInFlight in flight; requests no longer desired are
    //     cancelled;
    //  5. evicts tiles that are not desired, LRU first, so that each level holds at most tilesPerLevel (and the masks
    //     that are not desired);
    //  6. sends every removal of 5, then at most uploadsPerFrame tile additions by distance and the mask additions.
    // Without a render sink nothing is requested or sent
    void updateView(SpaceId space, const Vec3& camera);
    // Blocks until the desired set of this camera is resident and sent (the pyramid, then the tiles, then the masks,
    // repeated until nothing is missing; no maxInFlight or uploadsPerFrame limit; the calling thread runs queued jobs
    // too). Returns false on timeout (what is resident is still sent; no caller job starts after the deadline, a
    // running one is finished). Without a sink it fills the store only. A space without terrain returns true at once
    bool completeView(SpaceId space, const Vec3& camera, double timeoutSeconds = 120);
    // Nothing of this camera's desired set is missing (resident and sent): completeView would generate and send
    // nothing. True for a space without terrain; without a sink only the residency counts
    bool viewComplete(SpaceId space, const Vec3& camera) const;
    // D6: the fraction of (the pyramid + the desired tiles of this camera whose box is within radius) that have been
    // sent to the sink; 0 while the pyramid is not sent (or without a sink); 1 for a space without terrain
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
    };
    const Stats& stats() const { return statistics; }
    // The lazy StreamingPool exists (a sink alone starts none)
    bool poolCreated() const { return pool != nullptr; }

    // Engine-internal: the World's render sink (World::setRenderSink). A new sink receives the pyramid, the tiles and
    // the masks the store of the view space holds (through the send point); a removed sink drops the sent state
    void setSink(RenderSink* sink);
    // Tests: a space's store
    const TerrainStore& store(SpaceId space) const;

   private:
    struct Space
    {
        SpaceTemplate settings;
        uint64_t seed = 0;
        TerrainHeightSource source;
        std::unique_ptr<TerrainStore> store;
        // Per streamed level: the min / max height of each tile from the pyramid (row-major tiles), for the tile boxes
        std::vector<std::vector<std::pair<int32_t, int32_t>>> tileHeights;
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
    TerrainViewSettings view;
    RenderSink* sink = nullptr;
    std::unique_ptr<StreamingPool> pool;
    std::map<Key, uint64_t> requests;
    // The sink's set: active once a pyramid was sent
    bool setActive   = false;
    SpaceId setSpace = 0;
    Vulkan::TerrainLayout setLayout;
    std::set<RegionId> sentTiles;
    std::set<ChunkId> sentMasks;
    bool viewSpaceValid = false;
    SpaceId viewSpace   = 0;
    uint64_t frame      = 0;
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
    // Runs queued jobs on this thread until the deadline, then waits for the workers' jobs until it
    void runUntil(double deadlineSeconds);
    void updateStats(SpaceId space, const Desired* d);
    // The single send point (D3, E3): S22 merges its edits here
    void sendPyramid(SpaceId space, const Vulkan::TerrainPyramidData& pyramid);
    void sendTile(const RegionId& region, const Vulkan::TerrainTileData& tile);
    void sendGrassMask(const ChunkId& chunk, const Vulkan::GrassMaskData& mask);
    void computeTileHeights(Space& sp);
    std::string describe(const Key& key) const;
};

}  // namespace Engine
}  // namespace Saiga
