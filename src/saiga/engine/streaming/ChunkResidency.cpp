/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/streaming/ChunkResidency.h"

#include "saiga/engine/components/IslandState.h"
#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/StreamingAnchor.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/content/GeneratorList.h"
#include "saiga/engine/core/StreamingPool.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/streaming/Islands.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/Events.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace Saiga
{
namespace Engine
{
namespace
{
double nowSeconds()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The steps of the stall ring (the Engine tab's "last 600 steps")
constexpr size_t kStallRing = 600;

ChunkId chunkId(SpaceId space, int64_t x, int64_t z)
{
    ChunkId c;
    c.space = space;
    c.x     = int32_t(x);
    c.z     = int32_t(z);
    return c;
}

// The chunk index range [lo, hi] of the squares within distance r of v along one axis, clamped to [0, n)
void chunkRange(double v, double r, int64_t n, int64_t& lo, int64_t& hi)
{
    auto clampIndex = [n](double x) { return int64_t(std::min(std::max(x, -1.0), double(n))); };
    lo              = std::max<int64_t>(0, clampIndex(std::ceil((v - r) / 256) - 1));
    hi              = std::min<int64_t>(n - 1, clampIndex(std::floor((v + r) / 256)));
}

// A residency job: the request named in an exception (the pool's drain rethrows it; fatal in TerrainSystem)
struct JobResult
{
    std::shared_ptr<const ChunkBase> base;
    std::shared_ptr<const ChunkPlacement> placement;
};
}  // namespace

ChunkResidency::ChunkResidency(World& world) : w(world), shared(std::make_shared<Shared>()) {}

ChunkResidency::~ChunkResidency() = default;

// ---- queries

double ChunkResidency::chunkDistance(const ChunkId& c, const Vec3& p) const
{
    const double dx = std::max({double(c.x) * 256 - p.x(), 0.0, p.x() - double(c.x + 1) * 256});
    const double dz = std::max({double(c.z) * 256 - p.z(), 0.0, p.z() - double(c.z + 1) * 256});
    return std::sqrt(dx * dx + dz * dz);
}

ChunkState ChunkResidency::state(const ChunkId& chunk) const
{
    auto it = entries.find(chunk);
    return it == entries.end() ? ChunkState::None : it->second.state;
}

bool ChunkResidency::requested(const ChunkId& chunk) const
{
    return requestedSet.count(chunk) != 0;
}

bool ChunkResidency::ready(const ChunkId& chunk) const
{
    if (!requestedSet.count(chunk)) return false;
    auto it = entries.find(chunk);
    return it != entries.end() && it->second.state == ChunkState::Resident;
}

uint8_t ChunkResidency::partsReady(const ChunkId& chunk) const
{
    auto it = entries.find(chunk);
    return it == entries.end() ? uint8_t(0) : it->second.partsDone;
}

uint8_t ChunkResidency::partsWanted(const ChunkId& chunk) const
{
    if (chunk.space >= w.spaceCount()) return 0;
    uint8_t parts = 0;
    if (w.spaceHasTerrain(chunk.space)) parts |= PartBase;
    const GeneratorList::Entry* g = w.generators().find(w.generation().space(chunk.space).generatorId);
    if (g && !g->slots.empty()) parts |= PartPlacement;
    return parts;
}

bool ChunkResidency::kept(const ChunkId& chunk) const
{
    if (chunk.space >= w.spaceCount()) return false;
    const double k = w.spaceTemplate(chunk.space).keptDistance();
    for (const StreamingCentre& c : currentCentres)
        if (c.space == chunk.space && chunkDistance(chunk, c.position) <= k) return true;
    return false;
}

double ChunkResidency::islandProgress(SpaceId space, const Vec3& position) const
{
    if (space >= w.spaceCount() || !position.allFinite()) return 0;
    const SpaceTemplate& t = w.spaceTemplate(space);
    const double r         = t.simulationRadius;
    const int64_t n        = t.sizeChunks;
    // The position clamped into the space (the island of a position at the edge)
    const double size = t.sizeMeters();
    const Vec3 p(std::min(std::max(position.x(), 0.0), size), 0, std::min(std::max(position.z(), 0.0), size));
    int64_t x0, x1, z0, z1;
    chunkRange(p.x(), r, n, x0, x1);
    chunkRange(p.z(), r, n, z0, z1);
    int total = 0, done = 0;
    for (int64_t z = z0; z <= z1; ++z)
        for (int64_t x = x0; x <= x1; ++x)
        {
            const ChunkId c = chunkId(space, x, z);
            if (chunkDistance(c, p) > r) continue;
            total++;
            done += ready(c) ? 1 : 0;
        }
    return total == 0 ? 1.0 : double(done) / double(total);
}

int ChunkResidency::addListener(ResidencyListener listener)
{
    listeners.push_back({nextListenerId, std::move(listener)});
    return nextListenerId++;
}

void ChunkResidency::removeListener(int id)
{
    for (size_t i = 0; i < listeners.size(); ++i)
        if (listeners[i].first == id) listeners.erase(listeners.begin() + std::ptrdiff_t(i));
}

std::vector<ChunkStatus> ChunkResidency::chunkStatuses() const
{
    std::vector<ChunkStatus> out;
    out.reserve(entries.size());
    for (const auto& [c, e] : entries)
    {
        ChunkStatus s;
        s.chunk       = c;
        s.state       = e.state;
        s.wanted      = e.wanted;
        s.ready       = e.partsDone;
        s.requested   = requestedSet.count(c) != 0;
        s.distance    = e.distance;
        s.jobInFlight = e.jobId != 0;
        out.push_back(s);
    }
    return out;
}

std::shared_ptr<const ChunkPlacement> ChunkResidency::heldPlacement(const ChunkId& chunk,
                                                                    uint16_t placementVersion) const
{
    auto it = entries.find(chunk);
    if (it == entries.end() || !it->second.placement) return nullptr;
    return it->second.placement->placementVersion == placementVersion ? it->second.placement : nullptr;
}

bool ChunkResidency::clientAwaitsMove(Entity root) const
{
    for (const ClientArrival& a : clientArrivals)
        if (a.root == root) return true;
    return false;
}

// ---- centres

std::vector<StreamingCentre> ChunkResidency::computeCentres() const
{
    std::vector<StreamingCentre> out;
    Registry& reg   = const_cast<Registry&>(w.registry());
    const bool host = w.isHost();
    std::vector<uint32_t> seen;
    auto addRoot = [&](Entity e, bool needOwned)
    {
        const Entity r = w.rootOf(e);
        if (!r.valid() || std::find(seen.begin(), seen.end(), r.index) != seen.end()) return;
        // A client's own player roots only (the replicated ones of other peers are no centres; R-NET-1)
        if (needOwned && !host && w.ownerOf(r) != w.localPeer()) return;
        seen.push_back(r.index);
        const Transform& t = reg.read<Transform>(r);
        if (!t.position().allFinite() || t.space() >= w.spaceCount()) return;
        StreamingCentre c;
        c.space    = t.space();
        c.position = t.position();
        c.entity   = r;
        out.push_back(c);
    };
    reg.view<const Player>().each([&](Entity e, const Player&) { addRoot(e, true); });
    reg.view<const StreamingAnchor>().each([&](Entity e, const StreamingAnchor&) { addRoot(e, false); });
    // The targets of pending player arrivals (other roots' arrivals are no centres, G12)
    reg.view<const PendingTransition>().each(
        [&](Entity e, const PendingTransition& p)
        {
            if (reg.has<Parent>(e) || !w.isPlayerRoot(e)) return;
            if (!host && w.ownerOf(e) != w.localPeer()) return;
            if (p.target >= w.spaceCount() || !p.position.allFinite()) return;
            StreamingCentre c;
            c.space    = p.target;
            c.position = p.position;
            c.entity   = e;
            c.arrival  = true;
            out.push_back(c);
        });
    // Client (G4): a target stays a centre until the move's OwnerCommand is applied
    for (const ClientArrival& a : clientArrivals)
    {
        if (!reg.alive(a.root) || reg.has<PendingTransition>(a.root) || a.target >= w.spaceCount()) continue;
        StreamingCentre c;
        c.space    = a.target;
        c.position = a.position;
        c.entity   = a.root;
        c.arrival  = true;
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(),
              [](const StreamingCentre& a, const StreamingCentre& b)
              {
                  if (a.space != b.space) return a.space < b.space;
                  if (a.entity.index != b.entity.index) return a.entity.index < b.entity.index;
                  return a.arrival < b.arrival;
              });
    return out;
}

bool ChunkResidency::centresChanged() const
{
    return computeCentres() != currentCentres;
}

void ChunkResidency::updateClientArrivals()
{
    if (w.isHost()) return;
    const Registry& reg = w.registry();
    // The owned player roots with a pending arrival (replicated, G13)
    const_cast<Registry&>(reg).view<const PendingTransition>().each(
        [&](Entity e, const PendingTransition& p)
        {
            if (reg.has<Parent>(e) || w.ownerOf(e) != w.localPeer() || !w.isPlayerRoot(e)) return;
            for (ClientArrival& a : clientArrivals)
                if (a.root == e)
                {
                    a.target   = p.target;
                    a.position = p.position;
                    return;
                }
            ClientArrival a;
            a.root     = e;
            a.target   = p.target;
            a.position = p.position;
            a.counter  = w.replication().commandCounter(e);
            clientArrivals.push_back(a);
        });
    // A removed component: kept until the move's command is applied (its counter differs), or the root stands at the
    // target (the command was applied before the component arrived)
    clientArrivals.erase(std::remove_if(clientArrivals.begin(), clientArrivals.end(),
                                        [&](const ClientArrival& a)
                                        {
                                            if (!reg.alive(a.root)) return true;
                                            if (reg.has<PendingTransition>(a.root)) return false;
                                            if (w.replication().commandCounter(a.root) != a.counter) return true;
                                            const Transform& t = reg.read<Transform>(a.root);
                                            return t.space() == a.target && (t.position() - a.position).norm() < 1e-3;
                                        }),
                         clientArrivals.end());
}

// ---- the update

void ChunkResidency::requestPart(int limit)
{
    currentCentres     = computeCentres();
    statistics.centres = int(currentCentres.size());
    requestedSet.clear();
    requiredSet.clear();
    // The nearest distance of every chunk within the kept distance of a centre
    std::map<ChunkId, double> nearest;
    for (const StreamingCentre& c : currentCentres)
    {
        const SpaceTemplate& t = w.spaceTemplate(c.space);
        const double R = t.requiredDistance(), Q = t.requestedDistance(), K = t.keptDistance();
        int64_t x0, x1, z0, z1;
        chunkRange(c.position.x(), K, t.sizeChunks, x0, x1);
        chunkRange(c.position.z(), K, t.sizeChunks, z0, z1);
        for (int64_t z = z0; z <= z1; ++z)
            for (int64_t x = x0; x <= x1; ++x)
            {
                const ChunkId id = chunkId(c.space, x, z);
                const double d   = chunkDistance(id, c.position);
                if (d > K) continue;
                auto [it, added] = nearest.emplace(id, d);
                if (!added) it->second = std::min(it->second, d);
                if (d <= Q) requestedSet.insert(id);
                if (d <= R) requiredSet.insert(id);
            }
    }
    // Unloads: beyond the kept distance of every centre (G2: jobs cancelled, the tile unpinned, the placement released)
    for (auto it = entries.begin(); it != entries.end();)
    {
        if (nearest.count(it->first))
        {
            ++it;
            continue;
        }
        unload(it->first, it->second);
        it = entries.erase(it);
    }
    // New requests; the distances and the required flags
    for (const ChunkId& c : requestedSet)
    {
        auto [it, added] = entries.try_emplace(c);
        if (!added) continue;
        it->second.state  = ChunkState::Requested;
        it->second.wanted = partsWanted(c);
        statistics.requested++;
    }
    for (auto& [c, e] : entries)
    {
        e.distance = nearest[c];
        e.required = requiredSet.count(c) != 0;
    }
    checkVersions();
    if (w.streamingSettings().synchronous)
    {
        // Every requested chunk that is not complete is built inline (the cache's synchronous path), in ChunkId order
        for (const ChunkId& c : requestedSet)
        {
            Entry& e = entries[c];
            if (e.partsDone != e.wanted) buildInline(c, e);
        }
        return;
    }
    StreamingPool* pool = w.terrain().poolCreated() ? &w.terrain().streamingPool() : nullptr;
    // A queued prefetch job whose chunk became required is cancelled and resubmitted at the required priority (the pool
    // cannot re-prioritise; a running one is left alone)
    if (pool)
        for (auto& [c, e] : entries)
            if (e.jobId != 0 && !e.jobRequired && e.required && pool->cancel(e.jobId))
            {
                e.jobId = 0;
                shared->inFlight--;
            }
    // The missing parts of requested chunks by priority: required first, then prefetch, nearest first
    std::vector<std::tuple<int, int, ChunkId>> candidates;
    for (const ChunkId& c : requestedSet)
    {
        const Entry& e = entries[c];
        if (e.partsDone == e.wanted || e.jobId != 0) continue;
        candidates.push_back({e.required ? 0 : 1, int(std::min(e.distance, 1e6)), c});
    }
    std::sort(candidates.begin(), candidates.end());
    for (const auto& [kind, d, c] : candidates)
    {
        if (shared->inFlight >= limit) break;
        submit(c, entries[c], kind == 0);
    }
}

void ChunkResidency::checkVersions()
{
    // G6: a changed effective placement version (a load, a client record message, a test injection) drops a pending
    // result and requests again; a resident chunk drops its placement and goes back to Requested (the base stays)
    StreamingPool* pool = w.terrain().poolCreated() ? &w.terrain().streamingPool() : nullptr;
    for (auto& [c, e] : entries)
    {
        if (!(e.wanted & PartPlacement)) continue;
        const uint16_t v = w.effectivePlacementVersion(c);
        if ((e.partsDone & PartPlacement) && e.placement && e.placement->placementVersion != v)
        {
            e.placement.reset();
            e.partsDone = uint8_t(e.partsDone & ~PartPlacement);
            if (e.state == ChunkState::Resident) e.state = ChunkState::Requested;
        }
        if (e.jobId != 0 && e.version != v)
        {
            // Its result is dropped by the serial (a new request follows); a queued job is cancelled now
            if (pool && pool->cancel(e.jobId)) shared->inFlight--;
            e.jobId  = 0;
            e.serial = ++nextSerial;
        }
    }
}

void ChunkResidency::submit(const ChunkId& chunk, Entry& e, bool required)
{
    const bool needBase        = (e.wanted & PartBase) && !(e.partsDone & PartBase);
    const bool needPlacement   = (e.wanted & PartPlacement) && !(e.partsDone & PartPlacement);
    e.serial                   = ++nextSerial;
    e.version                  = (e.wanted & PartPlacement) ? w.effectivePlacementVersion(chunk) : uint16_t(0);
    e.sequence                 = w.terrainEdits().sequence(chunk);
    const ContentSource source = w.generation().contentSource(chunk.space);
    const std::function<void(const ChunkId&)> hook = jobHookForTests;
    const std::shared_ptr<Shared> sh               = shared;
    const uint64_t generation                      = shared->generation;
    const uint64_t serial                          = e.serial;
    const uint16_t version                         = e.version;
    auto result                                    = std::make_shared<JobResult>();
    const std::string what = "chunk residency (space " + std::to_string(chunk.space) + ", chunk " +
                             std::to_string(chunk.x) + ", " + std::to_string(chunk.z) + ")";
    StreamingPool::Job job =
        [what, source, chunk, needBase, needPlacement, version, hook, result](const std::atomic<bool>&)
    {
        try
        {
            if (hook) hook(chunk);
            if (needBase) result->base = source.baseInJob(chunk);
            // The placement reads the base the job just generated from the cache (G17)
            if (needPlacement) result->placement = source.placementInJob(chunk, version);
        }
        catch (const std::exception& x)
        {
            throw std::runtime_error(what + ": " + x.what());
        }
    };
    StreamingPool::Completion done = [sh, generation, serial, chunk, result](bool cancelled)
    {
        // A dropAll since the submission (a load, replaceSources): nothing is delivered or counted
        if (sh->generation != generation) return;
        sh->inFlight--;
        Delivery d;
        d.chunk     = chunk;
        d.serial    = serial;
        d.base      = result->base;
        d.placement = result->placement;
        d.cancelled = cancelled;
        sh->deliveries.push_back(std::move(d));
    };
    // Required chunks before everything (-100000 + d), prefetch after the visual tiles and before the masks (G16)
    const int priority = required ? -100000 + int(std::min(e.distance, 1e6)) : 50000 + int(std::min(e.distance, 1e6));
    e.jobId            = w.terrain().streamingPool().submit(std::move(job), priority, std::move(done));
    e.jobRequired      = required;
    shared->inFlight++;
}

void ChunkResidency::buildInline(const ChunkId& chunk, Entry& e)
{
    if ((e.wanted & PartBase) && !(e.partsDone & PartBase))
    {
        e.base = w.generation().chunkBase(chunk);
        e.partsDone |= PartBase;
    }
    if ((e.wanted & PartPlacement) && !(e.partsDone & PartPlacement))
    {
        e.version   = w.effectivePlacementVersion(chunk);
        e.placement = w.generation().chunkPlacement(chunk, e.version);
        e.partsDone |= PartPlacement;
    }
    e.jobId = 0;
    statistics.syncBuilt++;
}

void ChunkResidency::applyDeliveries()
{
    if (shared->deliveries.empty()) return;
    std::vector<Delivery> list;
    list.swap(shared->deliveries);
    std::stable_sort(list.begin(), list.end(), [](const Delivery& a, const Delivery& b) { return a.chunk < b.chunk; });
    for (Delivery& d : list)
    {
        auto it = entries.find(d.chunk);
        if (d.cancelled || it == entries.end() || it->second.serial != d.serial)
        {
            // A cancelled running job (an unload, a version change), an unloaded chunk, or one requested again since
            // (another serial)
            statistics.resultsDropped++;
            continue;
        }
        Entry& e = it->second;
        e.jobId  = 0;
        if (d.base)
        {
            e.base = d.base;
            e.partsDone |= PartBase;
        }
        if (d.placement)
        {
            if (d.placement->placementVersion != w.effectivePlacementVersion(d.chunk))
                statistics.resultsDropped++;
            else
            {
                e.placement = d.placement;
                e.partsDone |= PartPlacement;
            }
        }
        // G6: the result is base content; the modification state is read at every sample, so a result that arrives
        // after an edit is applied together with it (counted)
        if (w.terrainEdits().sequence(d.chunk) != e.sequence) statistics.appliedAfterModification++;
        statistics.applied++;
    }
}

void ChunkResidency::promote()
{
    for (auto& [c, e] : entries)
    {
        if (e.state != ChunkState::Requested || e.partsDone != e.wanted) continue;
        e.state = ChunkState::Resident;
        if (e.base && !e.pinned)
        {
            w.terrain().pinSimulationTile(c, e.base);
            e.pinned = true;
        }
    }
}

void ChunkResidency::notifyReadyChanges()
{
    std::vector<std::pair<ChunkId, bool>> flips;
    for (auto& [c, e] : entries)
    {
        const bool now = e.state == ChunkState::Resident && requestedSet.count(c);
        if (now == e.readyLast) continue;
        e.readyLast = now;
        flips.push_back({c, now});
    }
    for (const auto& [c, r] : flips)
        for (const auto& l : listeners)
            if (l.second.onReadyChanged) l.second.onReadyChanged(c, r);
}

void ChunkResidency::unload(const ChunkId& chunk, Entry& e)
{
    if (e.readyLast)
    {
        e.readyLast = false;
        for (const auto& l : listeners)
            if (l.second.onReadyChanged) l.second.onReadyChanged(chunk, false);
    }
    for (const auto& l : listeners)
        if (l.second.onUnload) l.second.onUnload(chunk);
    if (e.jobId != 0 && w.terrain().poolCreated() && w.terrain().streamingPool().cancel(e.jobId)) shared->inFlight--;
    if (e.pinned) w.terrain().unpinSimulationTile(chunk);
    statistics.unloaded++;
}

void ChunkResidency::countStalls()
{
    int stalled         = 0;
    const Registry& reg = w.registry();
    std::vector<uint32_t> seen;
    const_cast<Registry&>(reg).view<const Player>().each(
        [&](Entity e, const Player&)
        {
            const Entity r = w.rootOf(e);
            if (std::find(seen.begin(), seen.end(), r.index) != seen.end()) return;
            seen.push_back(r.index);
            if (!w.isHost() && w.ownerOf(r) != w.localPeer()) return;
            if (w.heldRoot(r)) return;
            const Transform& t = reg.read<Transform>(r);
            if (!w.chunkReady(t.space(), t.position().x(), t.position().z())) stalled++;
        });
    lastStalledPlayers = stalled;
    if (stalled > 0)
    {
        statistics.streamingStalls++;
        statistics.playerStallSteps += uint64_t(stalled);
    }
    stallRing.push_back(stalled > 0);
    recentStallCount += stalled > 0 ? 1 : 0;
    if (stallRing.size() > kStallRing)
    {
        recentStallCount -= stallRing.front() ? 1 : 0;
        stallRing.pop_front();
    }
}

void ChunkResidency::hostIslands(bool report)
{
    if (!w.isHost()) return;
    const int count    = Islands::count(w);
    statistics.islands = count;
    if (!report) return;
    const int limit = w.worldConfig().maxSimulationIslands;
    if (count > limit) statistics.islandLimitSteps++;
    const uint32_t reported = w.readGlobal<IslandState>().reportedCount;
    if (uint32_t(count) == reported) return;
    // The event and the state at the phase-7 flush together (an aborted phase drops both, G14)
    const bool emit      = count > limit && (int64_t(reported) <= limit || uint32_t(count) > reported);
    const uint32_t value = uint32_t(count);
    w.recordOrRun(
        [value, emit](World& world)
        {
            world.global<IslandState>().reportedCount = value;
            if (emit) world.events.emit(IslandLimitEvent{value});
        });
}

void ChunkResidency::updateCurrentStats()
{
    statistics.requestedChunks = int(requestedSet.size());
    int resident = 0, queued = 0;
    size_t bytes = 0;
    for (const auto& [c, e] : entries)
    {
        resident += e.state == ChunkState::Resident ? 1 : 0;
        queued += (e.state == ChunkState::Requested && e.jobId == 0 && e.partsDone != e.wanted) ? 1 : 0;
        if (e.base && e.base->heights) bytes += e.base->heights->size() * sizeof(int32_t);
        if (e.placement) bytes += e.placement->instances.size() * sizeof(TierOneInstance);
    }
    statistics.residentChunks = resident;
    statistics.queuedChunks   = queued;
    statistics.pendingJobs    = shared->inFlight;
    statistics.residentBytes  = bytes;
}

void ChunkResidency::update()
{
    const double start = nowSeconds();
    updateClientArrivals();
    requestPart(std::max(1, w.streamingSettings().maxInFlight));
    // Every completion, also the visual ones (main thread, outside parallelEach; G16)
    w.terrain().drainStreamingPool();
    applyDeliveries();
    promote();
    notifyReadyChanges();
    countStalls();
    hostIslands(true);
    updateCurrentStats();
    statistics.lastStreamingMs = (nowSeconds() - start) * 1000.0;
}

void ChunkResidency::refresh()
{
    updateClientArrivals();
    requestPart(std::max(1, w.streamingSettings().maxInFlight));
    promote();
    notifyReadyChanges();
    hostIslands(false);
    updateCurrentStats();
}

bool ChunkResidency::requiredComplete() const
{
    for (const ChunkId& c : requiredSet)
    {
        auto it = entries.find(c);
        if (it == entries.end() || it->second.partsDone != it->second.wanted) return false;
    }
    return true;
}

bool ChunkResidency::complete(double timeoutSeconds)
{
    const double deadline = nowSeconds() + timeoutSeconds;
    const int unlimited   = std::numeric_limits<int>::max();
    updateClientArrivals();
    requestPart(unlimited);
    w.terrain().drainStreamingPool();
    applyDeliveries();
    promote();
    notifyReadyChanges();
    while (!requiredComplete() && nowSeconds() < deadline)
    {
        StreamingPool& pool = w.terrain().streamingPool();
        if (!pool.runOneOnCaller())
        {
            if (pool.queued() == 0 && pool.running() == 0 && pool.finished() == 0 && shared->inFlight == 0 &&
                shared->deliveries.empty())
            {
                // Nothing can arrive: request again (a dropped result); stop if that submits nothing
                requestPart(unlimited);
                if (shared->inFlight == 0) break;
            }
            else
                pool.waitIdleFor(std::min(0.005, std::max(0.0, deadline - nowSeconds())));
        }
        w.terrain().drainStreamingPool();
        applyDeliveries();
        requestPart(unlimited);
        promote();
        notifyReadyChanges();
    }
    hostIslands(false);
    updateCurrentStats();
    return requiredComplete();
}

void ChunkResidency::dropAll()
{
    const bool pool = w.terrain().poolCreated();
    for (auto& [c, e] : entries)
    {
        if (e.jobId != 0 && pool) w.terrain().streamingPool().cancel(e.jobId);
        if (e.pinned) w.terrain().unpinSimulationTile(c);
    }
    entries.clear();
    requestedSet.clear();
    requiredSet.clear();
    currentCentres.clear();
    clientArrivals.clear();
    shared->generation++;
    shared->inFlight = 0;
    shared->deliveries.clear();
    updateCurrentStats();
}

}  // namespace Engine
}  // namespace Saiga
