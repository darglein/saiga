/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/ContentRenderSystem.h"

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/ContentConvert.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/content/Tombstones.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/core/StreamingPool.h"
#include "saiga/engine/sync/RenderSink.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Engine
{
namespace
{
// The distance in the xz plane from a point to a chunk's square
double chunkDistance(const ChunkId& c, const Vec3& p)
{
    const double x0 = c.x * chunkSize, z0 = c.z * chunkSize;
    const double dx = std::max({x0 - p.x(), 0.0, p.x() - (x0 + chunkSize)});
    const double dz = std::max({z0 - p.z(), 0.0, p.z() - (z0 + chunkSize)});
    return std::sqrt(dx * dx + dz * dz);
}

uint64_t g_requestCounter = 0;
uint64_t nextCounter()
{
    // Main thread only (the World's content system); unique across the systems of a process
    return ++g_requestCounter;
}
}  // namespace

ContentRenderSystem::ContentRenderSystem(World& world) : w(world) {}

ContentRenderSystem::~ContentRenderSystem() = default;

void ContentRenderSystem::setSettings(const ContentRenderSettings& settings)
{
    SAIGA_ASSERT(settings.centres == ContentCentres::None || sink,
                 "ContentRenderSettings: content render centres need a render sink (World::setRenderSink first)");
    if (settings.centres != config.centres) removeAll();
    config = settings;
}

void ContentRenderSystem::setSink(RenderSink* s)
{
    removeAll();
    sink = s;
    // A mode needs a sink: without one the content render is off
    if (!sink) config.centres = ContentCentres::None;
}

void ContentRenderSystem::dropAll()
{
    removeAll();
}

bool ContentRenderSystem::spaceHasContent(SpaceId space) const
{
    if (space >= w.spaceCount()) return false;
    const SpaceGeneration& g = w.generation().space(space);
    if (!w.generators().find(g.generatorId)->slots.empty()) return true;
    for (const auto& p : g.prefabs)
        if (p && !p->statics.empty()) return true;
    return false;
}

double ContentRenderSystem::fullRadius(SpaceId space) const
{
    return w.spaceTemplate(space).fullRadius;
}

void ContentRenderSystem::chunksAround(SpaceId space, const Vec3& centre, double r, std::set<ChunkId>& out) const
{
    if (!spaceHasContent(space)) return;
    const int32_t size = w.spaceTemplate(space).sizeChunks;
    const int64_t x0   = std::max<int64_t>(0, int64_t(std::floor((centre.x() - r) / chunkSize)));
    const int64_t x1   = std::min<int64_t>(size - 1, int64_t(std::floor((centre.x() + r) / chunkSize)));
    const int64_t z0   = std::max<int64_t>(0, int64_t(std::floor((centre.z() - r) / chunkSize)));
    const int64_t z1   = std::min<int64_t>(size - 1, int64_t(std::floor((centre.z() + r) / chunkSize)));
    for (int64_t z = z0; z <= z1; ++z)
        for (int64_t x = x0; x <= x1; ++x)
        {
            ChunkId c;
            c.space = space;
            c.x     = int32_t(x);
            c.z     = int32_t(z);
            if (chunkDistance(c, centre) <= r) out.insert(c);
        }
}

Vulkan::MeshId ContentRenderSystem::meshOf(const std::string& generator, uint32_t slot, const std::string& mesh)
{
    const Vulkan::MeshId id = mesh.empty() ? Vulkan::MeshId(0) : w.findMesh(mesh);
    if (id == 0 && loggedMeshes.insert({generator, slot, mesh}).second)
    {
        statistics.unknownMeshes++;
        SAIGA_LOG("content", "generator '%s' slot %u: the mesh \"%s\" is not registered (its instances are not drawn)",
                  generator.c_str(), slot, mesh.c_str());
    }
    return id;
}

std::vector<Vulkan::StaticInstance> ContentRenderSystem::buildInstances(const ChunkId& chunk, uint16_t version,
                                                                        std::vector<InstanceKey>* keys)
{
    const SpaceGeneration& g = w.generation().space(chunk.space);
    const auto& slots        = w.generators().find(g.generatorId)->slots;
    std::vector<Vulkan::StaticInstance> out;
    if (keys) keys->clear();
    std::shared_ptr<const ChunkPlacement> placement;
    if (!slots.empty()) placement = w.generation().chunkPlacement(chunk, version);
    const std::vector<StaticPart>& statics = w.generation().staticList(chunk);
    out.reserve((placement ? placement->instances.size() : 0) + statics.size());
    auto add = [&](float x, float y, float z, uint16_t yaw, float scale, Vulkan::MeshId mesh, InstanceKey key)
    {
        Vulkan::StaticInstance si;
        si.position[0] = x;
        si.position[1] = y;
        si.position[2] = z;
        const Quat r   = contentRotation(yaw);
        si.setRotation(quat(float(r.w()), float(r.x()), float(r.y()), float(r.z())));
        si.mesh  = mesh;
        si.scale = mesh == 0 ? 0.0f : scale;
        si.key   = uint32_t(key.generatorSlot) << 24 | (key.counter & 0xFFFFFFu);
        out.push_back(si);
        if (keys) keys->push_back(key);
    };
    if (placement)
        for (const TierOneInstance& t : placement->instances)
        {
            const GeneratorSlot& slot = slots[t.key.generatorSlot];
            const std::string mesh    = t.meshSlot < slot.meshes.size() ? slot.meshes[t.meshSlot] : std::string();
            // Exact in float: |x|, |z| < 2^16 in 1/256 m, |y| <= 2^23 height units
            add(float(t.x) / 256.0f, float(t.y) / 1024.0f, float(t.z) / 256.0f, t.yaw, float(contentScale(t.scale)),
                meshOf(g.generatorName, t.key.generatorSlot, mesh), t.key);
        }
    for (uint32_t i = 0; i < uint32_t(statics.size()); ++i)
    {
        const StaticPart& p = statics[i];
        InstanceKey k;
        k.generatorSlot  = 255;
        k.counter        = i;
        const int64_t lx = p.x - int64_t(chunk.x) * 65536, lz = p.z - int64_t(chunk.z) * 65536;
        add(float(lx) / 256.0f, float(p.y) / 1024.0f, float(lz) / 256.0f, p.yaw, 1.0f,
            meshOf(g.generatorName, 255, p.mesh), k);
    }
    return out;
}

void ContentRenderSystem::install(const ChunkId& chunk, ChunkSet& s, const ChunkPlacement& placement)
{
    (void)placement;
    std::vector<Vulkan::StaticInstance> instances = buildInstances(chunk, s.version, &s.keys);
    s.requested                                   = false;
    if (instances.empty())
    {
        s.empty = true;
        statistics.emptyChunks++;
        return;
    }
    // Per tier-1 index the local position and the re-snap flag (step 7 re-snaps from them)
    const SpaceGeneration& g = w.generation().space(chunk.space);
    const auto& slots        = w.generators().find(g.generatorId)->slots;
    s.localX.assign(instances.size(), 0);
    s.localZ.assign(instances.size(), 0);
    s.resnap.assign(instances.size(), 0);
    size_t i = 0;
    for (const TierOneInstance& t : placement.instances)
    {
        s.localX[i] = t.x;
        s.localZ[i] = t.z;
        s.resnap[i] = (slots[t.key.generatorSlot].flags & ReSnapOnEdit) ? 1 : 0;
        ++i;
    }
    s.original = instances;
    // The chunk's current tombstones and (an edited chunk) the re-snapped y
    s.applied.clear();
    if (const ChunkRecord* r = w.contentRecords().find(chunk))
        for (const InstanceKey& k : r->tombstones)
            if (std::binary_search(s.keys.begin(), s.keys.end(), k)) s.applied.push_back(k);
    const bool edited = w.terrainEdits().hasDeltas(chunk);
    for (size_t k = 0; k < instances.size(); ++k)
    {
        const bool dead = std::binary_search(s.applied.begin(), s.applied.end(), s.keys[k]);
        if (dead || (edited && s.resnap[k])) instances[k] = instanceNow(chunk, s, k, dead);
    }
    s.current = instances;
    RegionId region;
    region.space = chunk.space;
    region.level = 0;
    region.x     = chunk.x;
    region.z     = chunk.z;
    s.handle     = sink->addInstanceSet(region, instances, uint32_t(instances.size()));
    s.live       = true;
    s.dirty.clear();
    statistics.setsBuilt++;
}

void ContentRenderSystem::buildNow(const ChunkId& chunk, ChunkSet& s)
{
    s.counter                = nextCounter();
    s.version                = w.effectivePlacementVersion(chunk);
    const SpaceGeneration& g = w.generation().space(chunk.space);
    std::shared_ptr<const ChunkPlacement> p;
    if (!w.generators().find(g.generatorId)->slots.empty()) p = w.generation().chunkPlacement(chunk, s.version);
    install(chunk, s, p ? *p : ChunkPlacement());
}

void ContentRenderSystem::request(const ChunkId& chunk, ChunkSet& s, StreamingPool& pool)
{
    s.counter                = nextCounter();
    s.version                = w.effectivePlacementVersion(chunk);
    s.requested              = true;
    const SpaceGeneration& g = w.generation().space(chunk.space);
    const uint64_t counter   = s.counter;
    const uint16_t version   = s.version;
    auto done                = [this, chunk, counter, version](std::shared_ptr<const ChunkPlacement> p)
    {
        auto it = sets.find(chunk);
        if (it == sets.end() || it->second.counter != counter || it->second.version != version || it->second.live ||
            !sink || config.centres != ContentCentres::View)
        {
            statistics.resultsDropped++;
            if (it != sets.end() && it->second.counter == counter) it->second.requested = false;
            return;
        }
        // As the state is now: the current tombstones and re-snaps are applied before the upload
        install(chunk, it->second, p ? *p : ChunkPlacement());
    };
    statistics.poolJobs++;
    if (w.generators().find(g.generatorId)->slots.empty())
    {
        // Static parts only: nothing to generate
        done(nullptr);
        return;
    }
    if (jobHookForTests)
    {
        // A held job (tests): the hook runs on the worker before the generation
        auto hook                                    = jobHookForTests;
        ContentSource source                         = w.generation().contentSource(chunk.space);
        auto result                                  = std::make_shared<std::shared_ptr<const ChunkPlacement>>();
        std::shared_ptr<ContentCache> cache          = w.generation().sharedCache();
        std::shared_ptr<const SpaceGeneration> space = w.generation().spacePtr(chunk.space);
        pool.submit(
            [hook, cache, space, chunk, version, result](const std::atomic<bool>&)
            {
                hook();
                *result = generatePlacement(*cache, *space, chunk, version, true);
            },
            0,
            [result, done](bool cancelled)
            {
                if (!cancelled) done(*result);
            });
        return;
    }
    w.generation().contentSource(chunk.space).requestPlacement(pool, chunk, version, 0, done);
}

void ContentRenderSystem::removeSet(const ChunkId& chunk, ChunkSet& s)
{
    (void)chunk;
    if (s.live && sink) sink->removeInstanceSet(s.handle);
    if (s.live) statistics.setsRemoved++;
    s.live    = false;
    s.handle  = Vulkan::InstanceSetHandle();
    s.counter = nextCounter();
}

void ContentRenderSystem::removeAll()
{
    for (auto& [c, s] : sets) removeSet(c, s);
    sets.clear();
    haveViewSpace = false;
}

void ContentRenderSystem::removeFar(const std::vector<std::pair<SpaceId, Vec3>>& centres)
{
    for (auto it = sets.begin(); it != sets.end();)
    {
        bool keep = false;
        for (const auto& [space, p] : centres)
            if (space == it->first.space && chunkDistance(it->first, p) <= fullRadius(space) + chunkSize)
            {
                keep = true;
                break;
            }
        if (keep)
        {
            ++it;
            continue;
        }
        removeSet(it->first, it->second);
        it = sets.erase(it);
    }
}

void ContentRenderSystem::playersPass()
{
    if (config.centres != ContentCentres::Players) return;
    SAIGA_ASSERT(sink, "ContentRenderSystem: the Players centres need a render sink");
    statistics.passes++;
    std::vector<std::pair<SpaceId, Vec3>> centres;
    w.viewAll<const Transform, const Player>().each([&](Entity, const Transform& t, const Player&)
                                                    { centres.push_back({t.space(), t.position()}); });
    removeFar(centres);
    drainResnaps();
    std::set<ChunkId> desired;
    for (const auto& [space, p] : centres) chunksAround(space, p, fullRadius(space), desired);
    for (const ChunkId& c : desired)
    {
        ChunkSet& s = sets[c];
        if (s.live || s.empty) continue;
        buildNow(c, s);
    }
}

void ContentRenderSystem::viewUpdate(SpaceId space, const Vec3& camera, bool complete)
{
    if (config.centres != ContentCentres::View) return;
    SAIGA_ASSERT(sink, "ContentRenderSystem: the View centres need a render sink");
    statistics.passes++;
    haveViewSpace = true;
    viewSpace     = space;
    // D5: the sets of other spaces go on a switch
    removeFar({{space, camera}});
    drainResnaps();
    std::set<ChunkId> desired;
    chunksAround(space, camera, fullRadius(space), desired);
    std::vector<std::pair<double, ChunkId>> order;
    for (const ChunkId& c : desired) order.push_back({chunkDistance(c, camera), c});
    std::sort(order.begin(), order.end());
    int budget = std::max(0, config.setsPerFrame);
    for (const auto& [d, c] : order)
    {
        ChunkSet& s = sets[c];
        if (s.live || s.empty) continue;
        if (complete)
            buildNow(c, s);
        else if (!s.requested && budget > 0)
        {
            request(c, s, w.terrain().streamingPool());
            budget--;
        }
    }
}

void ContentRenderSystem::recordChanged(const ChunkId& chunk)
{
    auto it = sets.find(chunk);
    if (it == sets.end() || !sink) return;
    ChunkSet& s        = it->second;
    const uint16_t ver = w.effectivePlacementVersion(chunk);
    if (ver != s.version)
    {
        // Another placement: the set (or the request) is dropped and built again at the new version
        removeSet(chunk, s);
        s.version   = ver;
        s.requested = false;
        s.empty     = false;
        statistics.regenerated++;
        return;
    }
    if (!s.live) return;
    std::vector<InstanceKey> now;
    if (const ChunkRecord* r = w.contentRecords().find(chunk))
        for (const InstanceKey& k : r->tombstones)
            if (std::binary_search(s.keys.begin(), s.keys.end(), k)) now.push_back(k);
    auto indexOf = [&](const InstanceKey& k)
    { return uint32_t(std::lower_bound(s.keys.begin(), s.keys.end(), k) - s.keys.begin()); };
    for (const InstanceKey& k : now)
        if (!std::binary_search(s.applied.begin(), s.applied.end(), k))
        {
            const uint32_t i = indexOf(k);
            sink->removeInstance(s.handle, i);
            s.current[i].scale = 0;
            statistics.tombstonesHidden++;
        }
    for (const InstanceKey& k : s.applied)
        if (!std::binary_search(now.begin(), now.end(), k))
        {
            // The original instance (re-snapped if the chunk has deltas) reappears
            const uint32_t i = indexOf(k);
            s.current[i]     = instanceNow(chunk, s, i, false);
            sink->updateInstances(s.handle, i, {s.current[i]});
            statistics.instancesRestored++;
        }
    s.applied = std::move(now);
}

void ContentRenderSystem::terrainChanged(SpaceId space, const SampleRect& box)
{
    if (sets.empty() || box.empty()) return;
    // Split per chunk with a set (live or requested): one merged box per chunk, drained by the next update or pass
    const int64_t S = 256 / w.spaceTemplate(space).baseSpacing;
    for (auto& [c, s] : sets)
    {
        if (c.space != space || s.empty) continue;
        const int64_t i0 = int64_t(c.x) * S, k0 = int64_t(c.z) * S;
        if (box.i1 < i0 || box.i0 > i0 + S || box.k1 < k0 || box.k0 > k0 + S) continue;
        if (s.dirty.empty())
            s.dirty.push_back(box);
        else
        {
            SampleRect& d = s.dirty.front();
            d.i0          = std::min(d.i0, box.i0);
            d.k0          = std::min(d.k0, box.k0);
            d.i1          = std::max(d.i1, box.i1);
            d.k1          = std::max(d.k1, box.k1);
        }
    }
}

Vulkan::StaticInstance ContentRenderSystem::instanceNow(const ChunkId& chunk, const ChunkSet& s, size_t i,
                                                        bool tombstoned) const
{
    Vulkan::StaticInstance v = s.original[i];
    // R-CONTENT-1: the integer surface rule over the edited samples, exact in float
    if (i < s.resnap.size() && s.resnap[i] && w.spaceHasTerrain(chunk.space))
        v.position[1] = float(tierOneSurface(w, chunk, s.localX[i], s.localZ[i])) / 1024.0f;
    if (tombstoned) v.scale = 0;
    return v;
}

void ContentRenderSystem::drainResnaps()
{
    for (auto& [c, s] : sets)
    {
        if (s.dirty.empty()) continue;
        if (!s.live)
        {
            // A set built later re-snaps at creation
            s.dirty.clear();
            continue;
        }
        const SampleRect box = s.dirty.front();
        s.dirty.clear();
        const int64_t cell = int64_t(w.spaceTemplate(c.space).baseSpacing) * 256;
        // The box grown by one sample, in 1/256 m of the space
        const int64_t x0 = (box.i0 - 1) * cell, x1 = (box.i1 + 1) * cell;
        const int64_t z0 = (box.k0 - 1) * cell, z1 = (box.k1 + 1) * cell;
        size_t first = s.current.size(), last = 0;
        for (size_t i = 0; i < s.resnap.size(); ++i)
        {
            if (!s.resnap[i]) continue;
            const int64_t gx = int64_t(c.x) * 65536 + s.localX[i], gz = int64_t(c.z) * 65536 + s.localZ[i];
            if (gx < x0 || gx > x1 || gz < z0 || gz > z1) continue;
            const bool dead                = std::binary_search(s.applied.begin(), s.applied.end(), s.keys[i]);
            const Vulkan::StaticInstance v = instanceNow(c, s, i, dead);
            if (v.position[1] == s.current[i].position[1]) continue;
            s.current[i] = v;
            first        = std::min(first, i);
            last         = std::max(last, i);
        }
        if (first > last) continue;
        sink->updateInstances(s.handle, uint32_t(first),
                              std::vector<Vulkan::StaticInstance>(s.current.begin() + std::ptrdiff_t(first),
                                                                  s.current.begin() + std::ptrdiff_t(last + 1)));
        statistics.resnapped += last - first + 1;
    }
}

double ContentRenderSystem::readiness(SpaceId space, const Vec3& camera, double radius) const
{
    if (config.centres != ContentCentres::View || !sink) return 1;
    std::set<ChunkId> desired;
    chunksAround(space, camera, std::min(radius, fullRadius(space)), desired);
    if (desired.empty()) return 1;
    size_t ready = 0;
    for (const ChunkId& c : desired)
    {
        auto it = sets.find(c);
        if (it != sets.end() && (it->second.live || it->second.empty)) ready++;
    }
    return double(ready) / double(desired.size());
}

Vulkan::InstanceSetHandle ContentRenderSystem::setOf(const ChunkId& chunk) const
{
    auto it = sets.find(chunk);
    return it != sets.end() && it->second.live ? it->second.handle : Vulkan::InstanceSetHandle();
}

size_t ContentRenderSystem::liveSets() const
{
    size_t n = 0;
    for (const auto& kv : sets) n += kv.second.live ? 1 : 0;
    return n;
}

}  // namespace Engine
}  // namespace Saiga
