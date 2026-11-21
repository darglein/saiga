/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/TerrainSystem.h"

#include "saiga/engine/core/StreamingPool.h"
#include "saiga/engine/sync/RenderSink.h"
#include "saiga/engine/terrain/TerrainGeneration.h"
#include "saiga/engine/terrain/TerrainStore.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

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

// The distance from p to the box
double boxDistance(const Vec3& p, const Vec3& lo, const Vec3& hi)
{
    return (p.cwiseMax(lo).cwiseMin(hi) - p).norm();
}

RegionId regionOf(SpaceId space, int level, int32_t x, int32_t z)
{
    RegionId r;
    r.space = space;
    r.level = uint8_t(level);
    r.x     = x;
    r.z     = z;
    return r;
}

ChunkId chunkOf(SpaceId space, int32_t x, int32_t z)
{
    ChunkId c;
    c.space = space;
    c.x     = x;
    c.z     = z;
    return c;
}

constexpr int kindPyramid = 0, kindTile = 1, kindMask = 2;
}  // namespace

TerrainSystem::TerrainSystem(const std::vector<SpaceTemplate>& templates, const std::vector<uint64_t>& seeds)
{
    SAIGA_ASSERT(templates.size() == seeds.size());
    for (size_t i = 0; i < templates.size(); ++i)
    {
        Space s;
        s.settings = templates[i];
        s.seed     = seeds[i];
        s.source   = Engine::heightSource(s.settings, s.seed);
        s.store    = std::make_unique<TerrainStore>();
        spaces.push_back(std::move(s));
    }
}

TerrainSystem::~TerrainSystem()
{
    // First: no job outlives the stores its completion would touch
    pool.reset();
}

const SpaceTemplate& TerrainSystem::spaceTemplate(SpaceId space) const
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown space");
    return spaces[space].settings;
}

const TerrainHeightSource& TerrainSystem::heightSource(SpaceId space) const
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown space");
    return spaces[space].source;
}

uint64_t TerrainSystem::spaceSeed(SpaceId space) const
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown space");
    return spaces[space].seed;
}

const TerrainStore& TerrainSystem::store(SpaceId space) const
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown space");
    return *spaces[space].store;
}

Vulkan::TerrainLayout TerrainSystem::layout(SpaceId space) const
{
    const SpaceTemplate& t = spaceTemplate(space);
    Vulkan::TerrainLayout l;
    l.baseSpacing     = t.baseSpacing;
    l.sizeChunks      = t.sizeChunks;
    l.tilesPerLevel   = view.tilesPerLevel;
    l.grassMaskLayers = view.grassMaskLayers;
    return l;
}

const Vulkan::TerrainPyramidData* TerrainSystem::pyramid(SpaceId space) const
{
    return store(space).pyramid();
}

// ---- samples and heights

int32_t TerrainSystem::sample(SpaceId space, int64_t sx, int64_t sz) const
{
    const Space& s = spaces[space];
    if (!s.settings.hasTerrain()) return 0;
    // A resident level-0 tile (its samples are the source's: bit-equal, R-DET-1)
    const int64_t n1 = 256 / s.settings.baseSpacing;
    const int64_t tx = sx >= 0 ? sx / n1 : -1, tz = sz >= 0 ? sz / n1 : -1;
    if (tx >= 0 && tz >= 0 && tx < s.settings.sizeChunks && tz < s.settings.sizeChunks)
    {
        const TerrainStore::Tile* t = s.store->tile(regionOf(space, 0, int32_t(tx), int32_t(tz)));
        if (t)
        {
            const int64_t n = n1 + 1;
            return (*t->data.heights)[size_t((sz - tz * n1) * n + (sx - tx * n1))];
        }
    }
    const int64_t spacing = s.settings.baseSpacing;
    return s.source(sx * spacing, sz * spacing);
}

double TerrainSystem::height(SpaceId space, double x, double z) const
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown space");
    const Space& s = spaces[space];
    if (!s.settings.hasTerrain()) return 0;
    const double size = s.settings.sizeMeters();
    // Sample coordinates: exact for spacing 1 or 2
    const double spacing = s.settings.baseSpacing;
    const double gx      = std::min(std::max(x, 0.0), size) / spacing;
    const double gz      = std::min(std::max(z, 0.0), size) / spacing;
    const int64_t n      = int64_t(s.settings.sizeChunks) * 256 / s.settings.baseSpacing;
    // The last cell at the far edge (fx = 1: the sample at the edge exists, tiles share their borders)
    const int64_t ci  = std::min(int64_t(std::floor(gx)), n - 1);
    const int64_t cj  = std::min(int64_t(std::floor(gz)), n - 1);
    const double fx   = gx - double(ci);
    const double fz   = gz - double(cj);
    const int32_t h00 = sample(space, ci, cj), h11 = sample(space, ci + 1, cj + 1);
    // Only the samples of the containing triangle (A: fx >= fz uses h10, B: h01)
    int32_t h10 = 0, h01 = 0;
    if (fx >= fz)
        h10 = sample(space, ci + 1, cj);
    else
        h01 = sample(space, ci, cj + 1);
    return terrainSurfaceDouble(h00, h10, h01, h11, fx, fz) * kHeightUnit;
}

// ---- the pool

void TerrainSystem::ensurePool()
{
    if (!pool) pool = std::make_unique<StreamingPool>(std::max(1, view.workers));
}

void TerrainSystem::drainPool()
{
    if (!pool) return;
    try
    {
        pool->drain();
    }
    catch (const std::exception& e)
    {
        // A height source is pure and has no recoverable failure (§3): fatal, naming the request
        SAIGA_EXIT_ERROR(std::string("TerrainSystem: ") + e.what());
    }
}

std::string TerrainSystem::describe(const Key& key) const
{
    const auto [kind, space, level, x, z] = key;
    const std::string s                   = "space " + std::to_string(space);
    if (kind == kindPyramid) return "terrain pyramid " + s;
    if (kind == kindTile)
        return "terrain tile " + s + " region L" + std::to_string(level) + " (" + std::to_string(x) + ", " +
               std::to_string(z) + ")";
    return "grass mask " + s + " chunk (" + std::to_string(x) + ", " + std::to_string(z) + ")";
}

namespace
{
// The job body with its request named in an exception (drain rethrows it without knowing the request)
template <class F>
StreamingPool::Job namedJob(std::string what, F body)
{
    return [what = std::move(what), body = std::move(body)](const std::atomic<bool>&)
    {
        try
        {
            body();
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(what + ": " + e.what());
        }
        catch (...)
        {
            throw std::runtime_error(what + ": unknown exception");
        }
    };
}
}  // namespace

void TerrainSystem::requestPyramid(SpaceId space)
{
    const Key key(kindPyramid, space, 0, 0, 0);
    if (requests.count(key)) return;
    ensurePool();
    Space& sp                        = spaces[space];
    auto result                      = std::make_shared<Vulkan::TerrainPyramidData>();
    const TerrainHeightSource source = sp.source;
    const Vulkan::TerrainLayout lay  = layout(space);
    const std::optional<int32_t> sea = sp.settings.seaLevel;
    const uint64_t id                = pool->submit(
        namedJob(describe(key), [result, source, lay, sea]() { *result = generatePyramid(source, lay, sea); }), 0,
        [this, key, result, space](bool cancelled)
        {
            requests.erase(key);
            if (cancelled) return;
            Space& s = spaces[space];
            s.store->setPyramid(std::move(*result));
            computeTileHeights(s);
            statistics.generated++;
            statistics.pyramidsGenerated++;
        });
    requests[key] = id;
}

void TerrainSystem::requestTile(const RegionId& region, int priority)
{
    const Key key(kindTile, region.space, region.level, region.x, region.z);
    if (requests.count(key)) return;
    ensurePool();
    Space& sp                        = spaces[region.space];
    auto result                      = std::make_shared<Vulkan::TerrainTileData>();
    const TerrainHeightSource source = sp.source;
    const Vulkan::TerrainLayout lay  = layout(region.space);
    const uint64_t id                = pool->submit(
        namedJob(describe(key), [result, source, lay, region]() { *result = generateTile(source, lay, region); }),
        priority,
        [this, key, result, region](bool cancelled)
        {
            requests.erase(key);
            if (cancelled) return;
            spaces[region.space].store->addTile(region, std::move(*result), frame);
            statistics.generated++;
        });
    requests[key] = id;
}

void TerrainSystem::requestMask(const ChunkId& chunk, int priority)
{
    const Key key(kindMask, chunk.space, 0, chunk.x, chunk.z);
    if (requests.count(key)) return;
    const TerrainStore::Tile* tile = spaces[chunk.space].store->tile(regionOf(chunk.space, 0, chunk.x, chunk.z));
    if (!tile) return;
    ensurePool();
    auto result                      = std::make_shared<Vulkan::GrassMaskData>();
    const Vulkan::TerrainTileData t0 = tile->data;
    const Vulkan::TerrainLayout lay  = layout(chunk.space);
    const std::optional<int32_t> sea = spaces[chunk.space].settings.seaLevel;
    const double minNormalY = view.grassMinNormalY, minAboveSea = view.grassMinAboveSea;
    const uint64_t generation = maskGeneration;
    const uint64_t id         = pool->submit(namedJob(describe(key), [result, t0, lay, sea, minNormalY, minAboveSea]()
                                                      { *result = generateGrassMask(t0, lay, sea, minNormalY, minAboveSea); }),
                                             priority,
                                             [this, key, result, chunk, generation](bool cancelled)
                                             {
                                         requests.erase(key);
                                         if (cancelled || generation != maskGeneration) return;
                                         spaces[chunk.space].store->addMask(chunk, std::move(*result));
                                         statistics.generated++;
                                     });
    requests[key]             = id;
}

// ---- the view

void TerrainSystem::computeTileHeights(Space& sp)
{
    const Vulkan::TerrainPyramidData* p = sp.store->pyramid();
    if (!p) return;
    const int P = p->layout.pyramidSamplesPerSide();
    const int S = p->layout.streamedLevels();
    sp.tileHeights.assign(size_t(S), {});
    for (int k = 0; k < S; ++k)
    {
        const int W = p->layout.tilesPerSide(k);
        const int s = 8 << k;  // pyramid cells per tile side
        auto& level = sp.tileHeights[size_t(k)];
        level.resize(size_t(W) * size_t(W));
        for (int tz = 0; tz < W; ++tz)
            for (int tx = 0; tx < W; ++tx)
            {
                int32_t lo = kTerrainHeightLimit, hi = -kTerrainHeightLimit;
                for (int j = 0; j <= s; ++j)
                    for (int i = 0; i <= s; ++i)
                    {
                        const int32_t v = (*p->heights)[size_t(tz * s + j) * size_t(P) + size_t(tx * s + i)];
                        lo              = std::min(lo, v);
                        hi              = std::max(hi, v);
                    }
                level[size_t(tz) * size_t(W) + size_t(tx)] = {lo, hi};
            }
    }
}

TerrainSystem::Desired TerrainSystem::computeDesired(SpaceId space, const Vec3& camera) const
{
    Desired d;
    const Space& sp = spaces[space];
    if (!sp.store->pyramid() || sp.tileHeights.empty()) return d;
    const Vulkan::TerrainLayout L = layout(space);
    for (int k = 0; k < L.streamedLevels(); ++k)
    {
        const double range = view.residencyRange0 * std::ldexp(1.0, k);
        const double size  = std::ldexp(256.0, k);
        const int W        = L.tilesPerSide(k);
        const int x0       = std::max(0, int(std::floor((camera.x() - range) / size)));
        const int x1       = std::min(W - 1, int(std::floor((camera.x() + range) / size)));
        const int z0       = std::max(0, int(std::floor((camera.z() - range) / size)));
        const int z1       = std::min(W - 1, int(std::floor((camera.z() + range) / size)));
        auto& level        = d.tiles[size_t(k)];
        for (int tz = z0; tz <= z1; ++tz)
            for (int tx = x0; tx <= x1; ++tx)
            {
                const auto& h = sp.tileHeights[size_t(k)][size_t(tz) * size_t(W) + size_t(tx)];
                const Vec3 lo(tx * size, double(h.first) * kHeightUnit - 64, tz * size);
                const Vec3 hi((tx + 1) * size, double(h.second) * kHeightUnit + 64, (tz + 1) * size);
                const double dist = boxDistance(camera, lo, hi);
                if (dist <= range) level.push_back({dist, regionOf(space, k, tx, tz)});
            }
        std::stable_sort(level.begin(), level.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        if (int(level.size()) > L.tilesPerLevel)
        {
            d.truncated[size_t(k)] = int(level.size()) - L.tilesPerLevel;
            level.resize(size_t(L.tilesPerLevel));
        }
        for (const auto& t : level) d.tileSet.insert(t.second);
    }
    // Grass masks: the chunks (with a desired level-0 tile) within the grass radius
    if (view.grassRadius > 0 && L.grassMaskLayers > 0)
    {
        for (const auto& t : d.tiles[0])
            if (t.first <= view.grassRadius) d.masks.push_back({t.first, chunkOf(space, t.second.x, t.second.z)});
        if (int(d.masks.size()) > L.grassMaskLayers)
        {
            d.masksTruncated = int(d.masks.size()) - L.grassMaskLayers;
            d.masks.resize(size_t(L.grassMaskLayers));
        }
        for (const auto& m : d.masks) d.maskSet.insert(m.second);
    }
    return d;
}

void TerrainSystem::markDesired(SpaceId space, const Desired& d)
{
    for (const RegionId& r : d.tileSet) spaces[space].store->markDesired(r, frame);
}

void TerrainSystem::cancelUndesired(SpaceId space, const Desired& d)
{
    if (!pool) return;
    for (auto it = requests.begin(); it != requests.end();)
    {
        const auto [kind, s, level, x, z] = it->first;
        bool keep                         = true;
        if (kind == kindTile) keep = s == space && d.tileSet.count(regionOf(s, level, x, z));
        if (kind == kindMask) keep = s == space && d.maskSet.count(chunkOf(s, x, z));
        // A queued job is dropped now; a running one reports the cancel in its completion (which erases it)
        if (!keep && pool->cancel(it->second))
            it = requests.erase(it);
        else
            ++it;
    }
}

void TerrainSystem::issueRequests(SpaceId space, const Desired& d, size_t limit)
{
    const TerrainStore& st = *spaces[space].store;
    // Missing desired tiles by distance / range ratio, coarse levels first at equal ratio
    std::vector<std::tuple<double, int, RegionId>> missingTiles;
    for (int k = 0; k < 8; ++k)
        for (const auto& t : d.tiles[size_t(k)])
        {
            if (st.tile(t.second) || requests.count(Key(kindTile, space, k, t.second.x, t.second.z))) continue;
            const double ratio = t.first / (view.residencyRange0 * std::ldexp(1.0, k));
            missingTiles.push_back({ratio, -k, t.second});
        }
    std::sort(missingTiles.begin(), missingTiles.end(), [](const auto& a, const auto& b)
              { return std::tie(std::get<0>(a), std::get<1>(a)) < std::tie(std::get<0>(b), std::get<1>(b)); });
    for (const auto& [ratio, negLevel, region] : missingTiles)
    {
        if (requests.size() >= limit) return;
        // Pyramid 0, tiles 1.. by ratio and level, masks after them
        requestTile(region, 1 + int(std::min(ratio, 1.0) * 1000) * 8 + (7 + negLevel));
    }
    for (const auto& m : d.masks)
    {
        if (requests.size() >= limit) return;
        if (st.mask(m.second)) continue;
        requestMask(m.second, 100000 + int(m.first));
    }
}

void TerrainSystem::evict(SpaceId space, const Desired& d)
{
    TerrainStore& st              = *spaces[space].store;
    const Vulkan::TerrainLayout L = layout(space);
    for (int k = 0; k < L.streamedLevels(); ++k)
    {
        int over = st.tilesAtLevel(k) - L.tilesPerLevel;
        if (over <= 0) continue;
        // Not desired, least recently desired first (a desired tile is never evicted: the desired set is truncated)
        std::vector<std::pair<uint64_t, RegionId>> candidates;
        for (const auto& [r, t] : st.tiles())
            if (r.level == k && !d.tileSet.count(r)) candidates.push_back({t.lastDesired, r});
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < candidates.size() && over > 0; ++i, --over)
        {
            st.removeTile(candidates[i].second);
            statistics.evicted++;
        }
    }
    // The masks that left the grass radius
    std::vector<ChunkId> gone;
    for (const auto& m : st.masks())
        if (!d.maskSet.count(m.first)) gone.push_back(m.first);
    for (const ChunkId& c : gone) st.removeMask(c);
}

void TerrainSystem::sendUpdates(SpaceId space, const Desired& d, int limit)
{
    if (!sink || !setActive || setSpace != space) return;
    const TerrainStore& st = *spaces[space].store;
    // Removals first (not budgeted: the sink never holds a tile the store dropped)
    for (auto it = sentTiles.begin(); it != sentTiles.end();)
    {
        if (!st.tile(*it))
        {
            sink->removeTerrainTile(*it);
            it = sentTiles.erase(it);
        }
        else
            ++it;
    }
    for (auto it = sentMasks.begin(); it != sentMasks.end();)
    {
        if (!st.mask(*it))
        {
            sink->removeGrassMask(*it);
            it = sentMasks.erase(it);
        }
        else
            ++it;
    }
    // Tile additions by distance
    std::vector<std::pair<double, RegionId>> add;
    for (int k = 0; k < 8; ++k)
        for (const auto& t : d.tiles[size_t(k)])
            if (st.tile(t.second) && !sentTiles.count(t.second)) add.push_back(t);
    std::stable_sort(add.begin(), add.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    int sent = 0;
    for (const auto& t : add)
    {
        if (limit >= 0 && sent >= limit) break;
        sendTile(t.second, st.tile(t.second)->data);
        sent++;
    }
    statistics.uploadsLastUpdate = sent;
    for (const auto& m : d.masks)
        if (st.mask(m.second) && !sentMasks.count(m.second)) sendGrassMask(m.second, *st.mask(m.second));
}

bool TerrainSystem::missing(SpaceId space, const Desired& d) const
{
    const TerrainStore& st = *spaces[space].store;
    for (const RegionId& r : d.tileSet)
        if (!st.tile(r) || (sink && !sentTiles.count(r))) return true;
    for (const ChunkId& c : d.maskSet)
        if (!st.mask(c) || (sink && !sentMasks.count(c))) return true;
    return false;
}

void TerrainSystem::sendPyramid(SpaceId space, const Vulkan::TerrainPyramidData& pyramid)
{
    // The data with the current layout (the heights are shared)
    Vulkan::TerrainPyramidData p = pyramid;
    p.layout                     = layout(space);
    p.seaLevel                   = spaces[space].settings.seaLevel;
    sink->setTerrainPyramid(space, p);
    setActive = true;
    setSpace  = space;
    setLayout = p.layout;
    sentTiles.clear();
    sentMasks.clear();
}

void TerrainSystem::sendTile(const RegionId& region, const Vulkan::TerrainTileData& tile)
{
    sink->setTerrainTile(region, tile);
    sentTiles.insert(region);
}

void TerrainSystem::sendGrassMask(const ChunkId& chunk, const Vulkan::GrassMaskData& mask)
{
    sink->setGrassMask(chunk, mask);
    sentMasks.insert(chunk);
}

void TerrainSystem::beginView(SpaceId space)
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown view space");
    const bool terrain = spaces[space].settings.hasTerrain();
    if (viewSpaceValid && viewSpace != space)
    {
        // The old space's streamed CPU tiles and masks go (its pyramid stays cached) and its requests are cancelled
        spaces[viewSpace].store->dropStreamed();
        if (pool)
            for (auto it = requests.begin(); it != requests.end();)
            {
                if (std::get<1>(it->first) == viewSpace && std::get<0>(it->first) != kindPyramid &&
                    pool->cancel(it->second))
                    it = requests.erase(it);
                else
                    ++it;
            }
    }
    viewSpace            = space;
    viewSpaceValid       = true;
    statistics.viewSpace = space;
    if (!sink) return;
    if (!terrain)
    {
        // The view space has no terrain: the sink drops its set (once)
        if (setActive) sink->clearTerrain();
        setActive = false;
        sentTiles.clear();
        sentMasks.clear();
        return;
    }
    const bool sameSet = setActive && setSpace == space && setLayout == layout(space);
    if (sameSet) return;
    // A new set: the sink drops the old one itself when it receives the pyramid
    setActive = false;
    sentTiles.clear();
    sentMasks.clear();
    if (const Vulkan::TerrainPyramidData* p = spaces[space].store->pyramid()) sendPyramid(space, *p);
}

void TerrainSystem::updateView(SpaceId space, const Vec3& camera)
{
    frame++;
    drainPool();
    if (!sink)
    {
        updateStats(space, nullptr);
        return;
    }
    beginView(space);
    if (!spaces[space].settings.hasTerrain())
    {
        updateStats(space, nullptr);
        return;
    }
    Space& sp = spaces[space];
    if (!sp.store->pyramid())
    {
        requestPyramid(space);
        statistics.uploadsLastUpdate = 0;
        updateStats(space, nullptr);
        return;
    }
    if (!setActive) sendPyramid(space, *sp.store->pyramid());
    const Desired d = computeDesired(space, camera);
    markDesired(space, d);
    cancelUndesired(space, d);
    issueRequests(space, d, size_t(std::max(1, view.maxInFlight)));
    evict(space, d);
    sendUpdates(space, d, std::max(0, view.uploadsPerFrame));
    updateStats(space, &d);
}

void TerrainSystem::runUntil(double deadline)
{
    if (!pool) return;
    // The deadline is checked before each caller job; a running one is finished (it cannot be interrupted)
    while (nowSeconds() < deadline && pool->runOneOnCaller())
    {
    }
    const double remaining = deadline - nowSeconds();
    if (remaining > 0) pool->waitIdleFor(remaining);
}

bool TerrainSystem::completeView(SpaceId space, const Vec3& camera, double timeoutSeconds)
{
    const double start    = nowSeconds();
    const double deadline = start + timeoutSeconds;
    frame++;
    drainPool();
    beginView(space);
    auto finish = [&](bool ok, const Desired* d)
    {
        statistics.lastCompleteMs = (nowSeconds() - start) * 1000.0;
        updateStats(space, d);
        return ok;
    };
    if (!spaces[space].settings.hasTerrain()) return finish(true, nullptr);
    Space& sp = spaces[space];
    // (a) the pyramid
    if (!sp.store->pyramid())
    {
        requestPyramid(space);
        runUntil(deadline);
        drainPool();
        if (!sp.store->pyramid()) return finish(false, nullptr);
    }
    if (sink && !setActive) sendPyramid(space, *sp.store->pyramid());
    const Desired d = computeDesired(space, camera);
    markDesired(space, d);
    cancelUndesired(space, d);
    for (;;)
    {
        // (b) every missing desired tile
        issueRequests(space, d, std::numeric_limits<size_t>::max());
        runUntil(deadline);
        drainPool();
        // (c) the masks whose level-0 tile is now stored
        issueRequests(space, d, std::numeric_limits<size_t>::max());
        runUntil(deadline);
        drainPool();
        // (d) eviction and every send
        evict(space, d);
        sendUpdates(space, d, -1);
        if (!missing(space, d)) return finish(true, &d);
        if (nowSeconds() >= deadline) return finish(false, &d);
    }
}

double TerrainSystem::readiness(SpaceId space, const Vec3& camera, double radius) const
{
    if (!spaceTemplate(space).hasTerrain()) return 1;
    if (!sink || !setActive || setSpace != space) return 0;
    const Desired d = computeDesired(space, camera);
    int total = 1, sent = 1;
    for (int k = 0; k < 8; ++k)
        for (const auto& t : d.tiles[size_t(k)])
        {
            if (t.first > radius) continue;
            total++;
            sent += sentTiles.count(t.second) ? 1 : 0;
        }
    return double(sent) / double(total);
}

size_t TerrainSystem::terrainUploadsPending() const
{
    return sink ? sink->terrainUploadsPending() : 0;
}

void TerrainSystem::setViewSettings(const TerrainViewSettings& settings)
{
    const bool thresholds =
        settings.grassMinNormalY != view.grassMinNormalY || settings.grassMinAboveSea != view.grassMinAboveSea;
    view = settings;
    if (thresholds)
    {
        // Every mask is dropped and regenerated (older results are dropped by their generation)
        maskGeneration++;
        for (Space& s : spaces) s.store->dropMasks();
        if (sink)
            for (const ChunkId& c : sentMasks) sink->removeGrassMask(c);
        sentMasks.clear();
        if (pool)
            for (auto it = requests.begin(); it != requests.end();)
            {
                if (std::get<0>(it->first) == kindMask && pool->cancel(it->second))
                    it = requests.erase(it);
                else
                    ++it;
            }
    }
}

void TerrainSystem::setSink(RenderSink* s)
{
    sink      = s;
    setActive = false;
    sentTiles.clear();
    sentMasks.clear();
    if (!sink || !viewSpaceValid || !spaces[viewSpace].settings.hasTerrain()) return;
    const TerrainStore& st = *spaces[viewSpace].store;
    if (!st.pyramid()) return;
    // What the store holds, through the send point
    sendPyramid(viewSpace, *st.pyramid());
    for (const auto& [r, t] : st.tiles()) sendTile(r, t.data);
    for (const auto& [c, m] : st.masks()) sendGrassMask(c, m);
}

void TerrainSystem::updateStats(SpaceId space, const Desired* d)
{
    Stats& s = statistics;
    s.levels.fill(LevelStats());
    s.masksDesired = s.masksTruncated = 0;
    const TerrainStore& st            = *spaces[space].store;
    for (int k = 0; k < 8; ++k)
    {
        s.levels[size_t(k)].resident = st.tilesAtLevel(k);
        if (d)
        {
            s.levels[size_t(k)].desired   = int(d->tiles[size_t(k)].size());
            s.levels[size_t(k)].truncated = d->truncated[size_t(k)];
        }
    }
    for (const RegionId& r : sentTiles) s.levels[r.level].sent++;
    if (d)
    {
        s.masksDesired   = int(d->masks.size());
        s.masksTruncated = d->masksTruncated;
    }
    s.masksResident = int(st.masks().size());
    s.masksSent     = int(sentMasks.size());
    s.pending       = int(requests.size());
    s.viewSpace     = space;
    size_t bytes    = 0;
    for (const Space& sp : spaces) bytes += sp.store->cpuBytes();
    // The results in flight (at most one per request; their size by kind)
    for (const auto& [key, id] : requests)
    {
        const Vulkan::TerrainLayout L = layout(std::get<1>(key));
        const size_t n = size_t(L.samplesPerSide()), P = size_t(L.pyramidSamplesPerSide()),
                     m = size_t(L.grassMaskSize());
        bytes += std::get<0>(key) == kindPyramid ? P * P * 4 : (std::get<0>(key) == kindTile ? n * n * 4 : m * m);
    }
    s.cpuBytes = bytes;
}

}  // namespace Engine
}  // namespace Saiga
