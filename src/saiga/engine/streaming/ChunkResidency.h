/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ChunkResidency (docs/engine_design.md §12.3; S24), owned by the World (World::residency()): the resident chunk set of
// a streaming world around its streaming centres (G4: player roots, StreamingAnchor roots, the targets of pending
// player arrivals; a client's own). Per chunk the wanted parts (Base: the level-0 base heights pinned in the space's
// TerrainStore, for spaces with terrain; Placement: the ChunkPlacement at the chunk's effective placement version, for
// generators with slots) are built by one residency job on the streaming pool (G17) or inline in synchronous mode,
// applied in phase 7 only (G5) and dropped beyond the kept distance (G3). A chunk wanting no part is resident in the
// update that requests it. Derived data: never saved, rebuilt by a load (dropAll, then World::completeResidency).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/Types.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/streaming/StreamingStats.h"
#include "saiga/export.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;
struct ChunkBase;
struct ChunkPlacement;

enum class ChunkState : uint8_t
{
    None,
    Requested,
    Resident
};
// The parts of a chunk's residency (S25 adds PartStored = 4 through addPart, S28 PartColliders = 8)
enum ResidencyPart : uint8_t
{
    PartBase      = 1,
    PartPlacement = 2,
    PartStored    = 4
};

// A residency part added by a consumer (S25: the chunk store's stored groups; S28 the colliders): wanted per chunk at
// its request; built inline in synchronous mode, else requested (done() on the main thread applies it, e.g. from a
// pool completion: the chunk becomes Resident only with every wanted part); dropped when the chunk unloads
struct ResidencyPartInfo
{
    uint8_t bit = 0;
    std::function<bool(const ChunkId&)> wanted;
    std::function<void(const ChunkId&)> build;
    // priority: the chunk's residency priority (required -100000 + d, prefetch 50000 + d)
    std::function<void(const ChunkId&, int priority, std::function<void()> done)> request;
    std::function<void(const ChunkId&)> drop;
};

// A streaming centre (G4): the root's world position (also a held player at its current position) or the target of a
// pending player arrival (arrival true)
struct StreamingCentre
{
    SpaceId space = 0;
    Vec3 position = Vec3::Zero();
    Entity entity;
    bool arrival = false;
    bool operator==(const StreamingCentre& o) const
    {
        return space == o.space && entity == o.entity && arrival == o.arrival && position == o.position;
    }
};

// Called synchronously inside update() / refresh() / complete(), in ChunkId order per kind. Listeners never write World
// state (they run inside update(), also between steps and inside a load): a consumer accumulates and acts in its own
// system. S25 adds onResident and onKeptChanged
struct ResidencyListener
{
    // Before the chunk's data is dropped (-> None)
    std::function<void(const ChunkId&)> onUnload;
    // ready() flipped (G7)
    std::function<void(const ChunkId&, bool)> onReadyChanged;
    // S25: the chunk became Resident
    std::function<void(const ChunkId&)> onResident;
    // S25: kept() flipped, defined over the kept discs of the centres (also chunks that were never requested)
    std::function<void(const ChunkId&, bool)> onKeptChanged;
};

// One chunk's residency for the tools (streaming_status)
struct ChunkStatus
{
    ChunkId chunk;
    ChunkState state = ChunkState::None;
    uint8_t wanted   = 0;
    uint8_t ready    = 0;
    bool requested   = false;
    double distance  = 0;  // to the nearest centre of its space
    bool jobInFlight = false;
};

class SAIGA_ENGINE_API ChunkResidency
{
   public:
    explicit ChunkResidency(World& world);
    ~ChunkResidency();
    ChunkResidency(const ChunkResidency&)            = delete;
    ChunkResidency& operator=(const ChunkResidency&) = delete;

    // Phase 7 (the streaming system, both roles): the centres, the requested set, unloads, submissions (or the inline
    // builds), the pool drained, the deliveries applied, promotions; then the stall counters and, on the host, the
    // islands and IslandLimitEvent
    void update();
    // The step-start refresh (G5): the request part and the promotion (no drain, no delivery), the island count only
    void refresh();
    // The centres differ from those of the last update / refresh (the step start's check)
    bool centresChanged() const;
    // Between steps (World::completeResidency): update() without the stall part and with the island count only, then
    // the pool runs on the caller until every required chunk of every centre has its parts ready; false at the
    // deadline (what is ready is applied)
    bool complete(double timeoutSeconds);

    ChunkState state(const ChunkId& chunk) const;
    // Within the requested distance of a centre of the last update (G3)
    bool requested(const ChunkId& chunk) const;
    // Requested and resident (G7)
    bool ready(const ChunkId& chunk) const;
    uint8_t partsReady(const ChunkId& chunk) const;
    // The parts a chunk wants (Base for a space with terrain, Placement for a generator with slots)
    uint8_t partsWanted(const ChunkId& chunk) const;
    // The centres of the last update, sorted by (space, entity index, arrival)
    const std::vector<StreamingCentre>& centres() const { return currentCentres; }
    // Within the kept distance of a current centre (pure, G3)
    bool kept(const ChunkId& chunk) const;
    // The fraction of the chunks intersecting the simulationRadius disc around position that are ready, in [0, 1]
    // (arrival readiness, G12)
    double islandProgress(SpaceId space, const Vec3& position) const;
    int addListener(ResidencyListener listener);
    void removeListener(int id);
    // S25: a residency part (its bit above PartPlacement); before the first request
    void addPart(ResidencyPartInfo part);
    // S26 (D18): a chunk whose parts are ready stays Requested until gate(chunk) holds (a streaming client: the host's
    // chunk snapshot is applied); re-checked in every update(), refresh() and complete() (complete() does not wait for
    // it). Null: no gate
    void setModificationGate(std::function<bool(const ChunkId&)> gate) { modificationGate = std::move(gate); }
    // S26: fn(chunk, state) for every Requested or Resident chunk of a space, in ChunkId order (a client's chunk
    // interest)
    void forEachChunk(SpaceId space, const std::function<void(const ChunkId&, ChunkState)>& fn) const;
    // A load, replaceSources: unpins every tile it pinned, cancels its queued jobs, bumps the generation (completions
    // of running jobs deliver nothing), every chunk to None; no listener called (consumers reset themselves)
    void dropAll();
    const StreamingStats& stats() const { return statistics; }
    StreamingStats& statsForUpdate() { return statistics; }
    // Steps with a stalled player among the last 600 steps (the Engine tab)
    int recentStalls() const { return recentStallCount; }
    // Every chunk entry in ChunkId order (streaming_status)
    std::vector<ChunkStatus> chunkStatuses() const;
    // The held placement of a resident or pending chunk at this placement version (null: none; G17: resident chunks
    // never regenerate when the cache evicts)
    std::shared_ptr<const ChunkPlacement> heldPlacement(const ChunkId& chunk, uint16_t placementVersion) const;
    // Client (G4): the owned player root's arrival target stays a centre until the move's OwnerCommand is applied; the
    // root is held while it is listed
    bool clientAwaitsMove(Entity root) const;
    // The player roots whose chunk is not ready on this peer in the last update (G11)
    int stalledPlayers() const { return lastStalledPlayers; }

    // Runs in the residency's job before it generates (tests: a latch)
    std::function<void(const ChunkId&)> jobHookForTests;

   private:
    struct Entry
    {
        ChunkState state  = ChunkState::None;
        uint8_t wanted    = 0;
        uint8_t partsDone = 0;
        bool required     = false;
        bool readyLast    = false;  // ready() as last notified
        double distance   = 0;
        // The request (G16): its serial, the placement version and the terrain sequence at the request (G6)
        uint64_t serial   = 0;
        uint16_t version  = 0;
        uint32_t sequence = 0;
        uint64_t jobId    = 0;  // 0: no job in flight
        bool jobRequired  = false;
        // S25: the added parts requested and not delivered yet, and the serial of their requests
        uint8_t partsRequested = 0;
        uint64_t partSerial    = 0;
        std::shared_ptr<const ChunkBase> base;
        std::shared_ptr<const ChunkPlacement> placement;
        bool pinned = false;
    };
    struct Delivery
    {
        ChunkId chunk;
        uint64_t serial = 0;
        std::shared_ptr<const ChunkBase> base;
        std::shared_ptr<const ChunkPlacement> placement;
        uint8_t parts = 0;  // S25: an added part's delivery (serial: the entry's partSerial)
    };
    // Shared with the jobs' completions (never the residency: the pool may outlive it)
    struct Shared
    {
        uint64_t generation = 0;
        int inFlight        = 0;
        int partsInFlight   = 0;  // S25: requests of added parts
        std::vector<Delivery> deliveries;
    };
    struct ClientArrival
    {
        Entity root;
        SpaceId target  = 0;
        Vec3 position   = Vec3::Zero();
        uint8_t counter = 0;
    };

    World& w;
    std::map<ChunkId, Entry> entries;
    std::set<ChunkId> requestedSet, requiredSet;
    // S25: the chunks within the kept distance of a centre (the kept discs), for onKeptChanged; the added parts
    std::set<ChunkId> keptSet;
    std::vector<ResidencyPartInfo> parts;
    std::function<bool(const ChunkId&)> modificationGate;
    void requestParts();
    std::vector<StreamingCentre> currentCentres;
    std::shared_ptr<Shared> shared;
    uint64_t nextSerial = 0;
    std::vector<std::pair<int, ResidencyListener>> listeners;
    int nextListenerId = 0;
    std::vector<ClientArrival> clientArrivals;
    StreamingStats statistics;
    std::deque<bool> stallRing;
    int recentStallCount   = 0;
    int lastStalledPlayers = 0;

    std::vector<StreamingCentre> computeCentres() const;
    void updateClientArrivals();
    // Steps 1-3 of update(): centres, requested set, unloads, submissions (limit: max jobs in flight) or inline builds
    void requestPart(int limit);
    void applyDeliveries();
    void promote();
    void notifyReadyChanges();
    void unload(const ChunkId& chunk, Entry& e);
    void submit(const ChunkId& chunk, Entry& e, bool required);
    void buildInline(const ChunkId& chunk, Entry& e);
    void checkVersions();
    void countStalls();
    void hostIslands(bool report);
    void updateCurrentStats();
    bool requiredComplete() const;
    double chunkDistance(const ChunkId& chunk, const Vec3& p) const;
};

}  // namespace Engine
}  // namespace Saiga
