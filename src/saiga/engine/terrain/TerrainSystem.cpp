/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/TerrainSystem.h"

#include "saiga/engine/core/Log.h"
#include "saiga/engine/core/StreamingPool.h"
#include "saiga/engine/sync/RenderSink.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainGeneration.h"
#include "saiga/engine/terrain/TerrainStore.h"
#include "saiga/engine/terrain/TerrainVisualPatch.h"
#include "saiga/vulkan/scene/TerrainTiles.h"

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

// floor(v) as an index clamped to [lo, hi] (in double before the conversion: no overflow for any finite v)
int clampedIndex(double v, int lo, int hi)
{
    return int(std::min(std::max(std::floor(v), double(lo)), double(hi)));
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

TerrainSystem::TerrainSystem(std::vector<TerrainSpace> list)
{
    for (TerrainSpace& t : list)
    {
        Space s;
        s.settings       = std::move(t.settings);
        s.seed           = t.seed;
        s.source         = std::move(t.source);
        s.hasTerrain     = t.hasTerrain && bool(s.source);
        s.generatorMasks = t.generatorMasks && t.content.valid();
        s.content        = std::move(t.content);
        s.grassRule      = t.grassRule;
        s.store          = std::make_unique<TerrainStore>();
        spaces.push_back(std::move(s));
    }
    visual = std::make_unique<TerrainVisualPatch>();
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

bool TerrainSystem::hasTerrain(SpaceId space) const
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown space");
    return spaces[space].hasTerrain;
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
    if (!s.hasTerrain) return 0;
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
    hookSamples.fetch_add(1, std::memory_order_relaxed);
    return s.source(sx * spacing, sz * spacing);
}

int32_t TerrainSystem::editedSample(SpaceId space, int64_t sx, int64_t sz) const
{
    const int32_t base = sample(space, sx, sz);
    if (!edits) return base;
    const int64_t v = int64_t(base) + edits->delta(space, sx, sz);
    return int32_t(std::min<int64_t>(std::max<int64_t>(v, -kTerrainHeightLimit), kTerrainHeightLimit));
}

double TerrainSystem::height(SpaceId space, double x, double z) const
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown space");
    const Space& s = spaces[space];
    // D1: a space without terrain is flat at 0 while it has no edits
    if (!s.hasTerrain && (!edits || edits->nonZeroSamples(space) == 0)) return 0;
    // A NaN coordinate has no point (it would survive the clamps below): 0; infinities clamp to the edges
    if (std::isnan(x) || std::isnan(z)) return 0;
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
    const int32_t h00 = editedSample(space, ci, cj), h11 = editedSample(space, ci + 1, cj + 1);
    // Only the samples of the containing triangle (A: fx >= fz uses h10, B: h01)
    int32_t h10 = 0, h01 = 0;
    if (fx >= fz)
        h10 = editedSample(space, ci + 1, cj);
    else
        h01 = editedSample(space, ci, cj + 1);
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
            statistics.generated++;
            statistics.pyramidsGenerated++;
        });
    requests[key] = id;
    jobsSubmitted++;
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
            // A job that finished before beginView left its space (its cancel came too late): not stored there
            if (cancelled || !viewSpaceValid || region.space != viewSpace) return;
            spaces[region.space].store->addTile(region, std::move(*result), frame);
            statistics.generated++;
        });
    requests[key] = id;
    jobsSubmitted++;
}

namespace
{
// An all-zero mask of the layout (a legacy space without a grass rule)
Vulkan::GrassMaskData zeroMask(const Vulkan::TerrainLayout& layout)
{
    Vulkan::GrassMaskData m;
    m.size    = 256 / layout.baseSpacing;
    m.density = std::make_shared<const std::vector<uint8_t>>(size_t(m.size) * size_t(m.size), uint8_t(0));
    return m;
}

Vulkan::GrassMaskData fromGenerator(const GrassMask& g)
{
    Vulkan::GrassMaskData m;
    m.size    = g.size;
    m.density = std::make_shared<const std::vector<uint8_t>>(g.density);
    return m;
}

// The generator mask where S20's rule (over the merged heights) allows grass (E3)
Vulkan::GrassMaskData andRule(const GrassMask& g, const Vulkan::GrassMaskData& rule)
{
    std::vector<uint8_t> d = g.density;
    for (size_t i = 0; i < d.size(); ++i)
        if (!rule.density || i >= rule.density->size() || (*rule.density)[i] == 0) d[i] = 0;
    Vulkan::GrassMaskData m;
    m.size    = g.size;
    m.density = std::make_shared<const std::vector<uint8_t>>(std::move(d));
    return m;
}
}  // namespace

uint16_t TerrainSystem::placementVersion(const ChunkId& chunk) const
{
    if (placementVersionOf) return placementVersionOf(chunk);
    return spaces[chunk.space].content.valid() ? spaces[chunk.space].content.space().current : uint16_t(1);
}

void TerrainSystem::requestMask(const ChunkId& chunk, int priority)
{
    const Key key(kindMask, chunk.space, 0, chunk.x, chunk.z);
    if (requests.count(key)) return;
    const TerrainStore::Tile* tile = spaces[chunk.space].store->tile(regionOf(chunk.space, 0, chunk.x, chunk.z));
    if (!tile) return;
    ensurePool();
    const Space& sp                       = spaces[chunk.space];
    auto result                           = std::make_shared<Vulkan::GrassMaskData>();
    const Vulkan::TerrainTileData t0      = tile->data;
    const Vulkan::TerrainLayout lay       = layout(chunk.space);
    const std::optional<int32_t> sea      = sp.settings.seaLevel;
    const std::optional<HookGrassRule> rl = sp.grassRule;
    const bool generator                  = sp.generatorMasks;
    const ContentSource content           = sp.content;
    const uint16_t version                = generator ? placementVersion(chunk) : uint16_t(0);
    const uint64_t generation             = maskGeneration;
    const std::function<void()> hook      = maskJobHookForTests;
    const uint64_t id =
        pool->submit(namedJob(describe(key),
                              [result, t0, lay, sea, rl, generator, content, chunk, version, hook]()
                              {
                                  if (hook) hook();
                                  // S23b (E3): a generator space's mask from its ChunkPlacement; a legacy space S20's
                                  // rule
                                  if (generator)
                                      *result = fromGenerator(content.placementInJob(chunk, version)->grass);
                                  else if (rl)
                                      *result = generateGrassMask(t0, lay, sea, rl->minNormalY, rl->minAboveSea);
                                  else
                                      *result = zeroMask(lay);
                              }),
                     priority,
                     [this, key, result, chunk, generation, generator, version](bool cancelled)
                     {
                         requests.erase(key);
                         if (cancelled || generation != maskGeneration || !viewSpaceValid || chunk.space != viewSpace)
                             return;
                         // A placement-version change meanwhile: dropped (requested again while desired)
                         if (generator && version != placementVersion(chunk)) return;
                         spaces[chunk.space].store->addMask(chunk, std::move(*result));
                         if (generator) maskVersions[chunk] = version;
                         statistics.generated++;
                     });
    requests[key] = id;
    jobsSubmitted++;
}

void TerrainSystem::recordChanged(const ChunkId& chunk)
{
    if (chunk.space >= spaces.size() || !spaces[chunk.space].generatorMasks) return;
    const uint16_t version = placementVersion(chunk);
    // The stored mask of another version goes (requested again while desired; the sink's is removed and re-sent)
    auto it = maskVersions.find(chunk);
    if (it != maskVersions.end() && it->second != version)
    {
        spaces[chunk.space].store->removeMask(chunk);
        maskVersions.erase(it);
        // The sink's copy is replaced even if the new mask is stored before the next send pass sees the gap
        // (completeView drains before it sends): the chunk does not count as sent
        if (sentMasks.count(chunk)) baseResend.insert(chunk);
    }
    // An edited chunk's mask includes the generator mask: recomputed
    auto e = visual->masks.find(chunk);
    if (e != visual->masks.end() && e->second.version != version)
    {
        visual->masks.erase(e);
        queueMask(chunk);
    }
}

// ---- the view

TerrainSystem::Desired TerrainSystem::computeDesired(SpaceId space, const Vec3& camera) const
{
    Desired d;
    const Space& sp = spaces[space];
    if (!sp.store->pyramid() || !camera.allFinite()) return d;
    const Vulkan::TerrainLayout L = layout(space);
    // TerrainPass's LOD distance (§12.2): the horizontal distance to the tile and the camera height above the surface
    // under it (one value per call), so the tiles the pass refines into are the desired ones at any relief
    const double hc = std::abs(camera.y() - height(space, camera.x(), camera.z()));
    for (int k = 0; k < L.streamedLevels(); ++k)
    {
        const double range = view.residencyRange0 * std::ldexp(1.0, k);
        const double size  = std::ldexp(256.0, k);
        const int W        = L.tilesPerSide(k);
        // The lower bounds: ceil - 1 also visits the tile that ends exactly at camera - range (its distance can be
        // range)
        const int x0 = clampedIndex(std::ceil((camera.x() - range) / size) - 1, 0, W);
        const int x1 = clampedIndex((camera.x() + range) / size, -1, W - 1);
        const int z0 = clampedIndex(std::ceil((camera.z() - range) / size) - 1, 0, W);
        const int z1 = clampedIndex((camera.z() + range) / size, -1, W - 1);
        auto& level  = d.tiles[size_t(k)];
        for (int tz = z0; tz <= z1; ++tz)
            for (int tx = x0; tx <= x1; ++tx)
            {
                const double dx   = std::max({0.0, tx * size - camera.x(), camera.x() - (tx + 1) * size});
                const double dz   = std::max({0.0, tz * size - camera.z(), camera.z() - (tz + 1) * size});
                const double dist = std::sqrt(dx * dx + dz * dz + hc * hc);
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
    // Grass masks: the chunks (with a desired level-0 tile) whose box is within the grass radius: the 3D distance (the
    // grass fades out by its 3D distance, GrassPass), the box from the pyramid's heights of the chunk +- 64 m, widened
    // by the deltas of an edited chunk (PR review: the base box of a raised chunk was too far for its grass)
    if (view.grassRadius > 0 && L.grassMaskLayers > 0)
    {
        const Vulkan::TerrainPyramidData& p = *sp.store->pyramid();
        const int P                         = L.pyramidSamplesPerSide();
        for (const auto& t : d.tiles[0])
        {
            if (t.first - hc > view.grassRadius) continue;  // horizontally out of the radius (dxz >= t.first - hc)
            int32_t lo = kTerrainHeightLimit, hi = -kTerrainHeightLimit;
            for (int j = 0; j <= 8; ++j)
                for (int i = 0; i <= 8; ++i)
                {
                    const int32_t v = (*p.heights)[size_t(t.second.z * 8 + j) * size_t(P) + size_t(t.second.x * 8 + i)];
                    lo              = std::min(lo, v);
                    hi              = std::max(hi, v);
                }
            int32_t dlo = 0, dhi = 0;
            if (const ChunkId c = chunkOf(space, t.second.x, t.second.z); editedChunk(c))
                edits->deltaRange(c, dlo, dhi);
            const Vec3 bmin(t.second.x * 256.0, (double(lo) + double(dlo)) * kHeightUnit - 64, t.second.z * 256.0);
            const Vec3 bmax((t.second.x + 1) * 256.0, (double(hi) + double(dhi)) * kHeightUnit + 64,
                            (t.second.z + 1) * 256.0);
            const double dist = (camera.cwiseMax(bmin).cwiseMin(bmax) - camera).norm();
            if (dist <= view.grassRadius) d.masks.push_back({dist, chunkOf(space, t.second.x, t.second.z)});
        }
        std::stable_sort(d.masks.begin(), d.masks.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
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
    if (requestsHeld) return;
    const TerrainStore& st = *spaces[space].store;
    // Missing desired tiles by distance / range ratio, coarse levels first at equal ratio
    std::vector<std::tuple<double, int, RegionId>> missingTiles;
    for (int k = 0; k < 8; ++k)
        for (const auto& t : d.tiles[size_t(k)])
        {
            if (st.tile(t.second) || requests.count(Key(kindTile, space, k, t.second.x, t.second.z))) continue;
            // A zero range desires only tiles at distance 0 (ratio 0, not 0 / 0)
            const double range = view.residencyRange0 * std::ldexp(1.0, k);
            const double ratio = range > 0 ? t.first / range : 0.0;
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
        // S22: a chunk with deltas gets its mask from merged heights (queued, capped per update); without a sink (the
        // store only, nothing is drawn) its base mask like any chunk's
        if (sink && editedChunk(m.second))
        {
            if (!readyMask(m.second)) queueMask(m.second);
            continue;
        }
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
        // Visual tiles only (S24: a pinned tile that is not visual is not in the budget)
        for (const auto& [r, t] : st.tiles())
            if (r.level == k && t.visual && !d.tileSet.count(r)) candidates.push_back({t.lastDesired, r});
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < candidates.size() && over > 0; ++i, --over)
        {
            st.removeTile(candidates[i].second);
            statistics.evicted++;
        }
    }
    // The masks that left the grass radius (S22: the edited ones too)
    std::vector<ChunkId> gone;
    for (const auto& m : st.masks())
        if (!d.maskSet.count(m.first)) gone.push_back(m.first);
    for (const ChunkId& c : gone)
    {
        st.removeMask(c);
        maskVersions.erase(c);
    }
    for (auto it = visual->masks.begin(); it != visual->masks.end();)
        if (!d.maskSet.count(it->first))
            it = visual->masks.erase(it);
        else
            ++it;
}

void TerrainSystem::sendUpdates(SpaceId space, const Desired& d, int limit)
{
    if (!sink || !setActive || setSpace != space) return;
    const TerrainStore& st = *spaces[space].store;
    // Removals first (not budgeted: the sink never holds a tile the store dropped, S24: or that is no longer visual, a
    // pinned tile that left the desired set; the atlas holds tilesPerLevel)
    for (auto it = sentTiles.begin(); it != sentTiles.end();)
    {
        const TerrainStore::Tile* tile = st.tile(*it);
        if (!tile || !tile->visual)
        {
            sink->removeTerrainTile(*it);
            it = sentTiles.erase(it);
        }
        else
            ++it;
    }
    for (auto it = sentMasks.begin(); it != sentMasks.end();)
    {
        if (!st.mask(*it) && !visual->masks.count(*it))
        {
            sink->removeGrassMask(*it);
            baseResend.erase(*it);
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
    {
        if (sentMasks.count(m.second) && !baseResend.count(m.second)) continue;
        if (const Vulkan::GrassMaskData* g = readyMask(m.second))
        {
            baseResend.erase(m.second);
            sendGrassMask(m.second, *g);
        }
    }
}

bool TerrainSystem::missing(SpaceId space, const Desired& d) const
{
    return missingCounts(space, d).total() > 0;
}

TerrainSystem::MissingCounts TerrainSystem::missingCounts(SpaceId space, const Desired& d) const
{
    MissingCounts m;
    const TerrainStore& st = *spaces[space].store;
    for (const RegionId& r : d.tileSet)
    {
        if (!st.tile(r))
            m.tilesNotStored++;
        else if (sink && !sentTiles.count(r))
            m.tilesNotSent++;
    }
    for (const ChunkId& c : d.maskSet)
    {
        // Without a sink the store's base mask (edited masks are made for a sink only)
        if (!(sink ? readyMask(c) : st.mask(c)))
            m.masksNotStored++;
        else if (sink && (!sentMasks.count(c) || baseResend.count(c)))
            m.masksNotSent++;
    }
    return m;
}

namespace
{
struct VisualTimer
{
    double& ms;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    explicit VisualTimer(double& ms) : ms(ms) {}
    ~VisualTimer()
    {
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};
}  // namespace

void TerrainSystem::sendPyramid(SpaceId space, const Vulkan::TerrainPyramidData& pyramid)
{
    // The data with the current layout (the heights are shared); S22: merged with the space's edits
    Vulkan::TerrainPyramidData p = pyramid;
    if (edits)
    {
        VisualTimer t(visualMs);
        Vulkan::TerrainPyramidData merged;
        if (TerrainVisualPatch::mergePyramid(*edits, space, pyramid, merged))
        {
            p = merged;
            statistics.mergedSends++;
        }
    }
    p.layout   = layout(space);
    p.seaLevel = spaces[space].settings.seaLevel;
    sink->setTerrainPyramid(space, p);
    setActive = true;
    setSpace  = space;
    setLayout = p.layout;
    sentTiles.clear();
    sentMasks.clear();
}

void TerrainSystem::sendTile(const RegionId& region, const Vulkan::TerrainTileData& tile)
{
    // S22: a merged copy if a chunk it covers has deltas (a result requested before an edit never undoes it)
    if (edits)
    {
        VisualTimer t(visualMs);
        Vulkan::TerrainTileData merged;
        if (TerrainVisualPatch::mergeTile(*edits, region, layout(region.space), tile, merged))
        {
            sink->setTerrainTile(region, merged);
            sentTiles.insert(region);
            statistics.mergedSends++;
            return;
        }
    }
    sink->setTerrainTile(region, tile);
    sentTiles.insert(region);
}

// ---- terrain edits (S22)

bool TerrainSystem::editedChunk(const ChunkId& chunk) const
{
    return edits && edits->hasDeltas(chunk);
}

const Vulkan::GrassMaskData* TerrainSystem::readyMask(const ChunkId& chunk) const
{
    if (editedChunk(chunk)) return visual->currentMask(chunk, edits->revision(chunk));
    return spaces[chunk.space].store->mask(chunk);
}

void TerrainSystem::queueMask(const ChunkId& chunk)
{
    if (std::find(maskQueue.begin(), maskQueue.end(), chunk) == maskQueue.end()) maskQueue.push_back(chunk);
}

int TerrainSystem::processMaskQueue(int limit)
{
    int submitted = 0;
    size_t i      = 0;
    for (; i < maskQueue.size(); ++i)
    {
        if (limit >= 0 && submitted >= limit) break;
        const ChunkId c = maskQueue[i];
        // Not wanted any more: dropped (a new job when it becomes desired again)
        if (!sink || !setActive || c.space != setSpace || !desiredMasks.count(c)) continue;
        if (editedChunk(c))
        {
            // In flight: its completion re-queues if the revision changed meanwhile
            if (editedJobs.count(c) || visual->currentMask(c, edits->revision(c))) continue;
            requestEditedMask(c);
            if (editedJobs.count(c)) submitted++;
            continue;
        }
        // The deltas returned to zero: the store's base mask replaces the edited one (requested if not resident)
        visual->masks.erase(c);
        if (sentMasks.count(c))
        {
            if (const Vulkan::GrassMaskData* b = spaces[c.space].store->mask(c))
                sendGrassMask(c, *b);
            else
            {
                baseResend.insert(c);
                requestMask(c, 100000);
            }
        }
    }
    maskQueue.erase(maskQueue.begin(), maskQueue.begin() + std::ptrdiff_t(i));
    return submitted;
}

void TerrainSystem::requestEditedMask(const ChunkId& chunk)
{
    Space& sp                      = spaces[chunk.space];
    const RegionId region          = regionOf(chunk.space, 0, chunk.x, chunk.z);
    const TerrainStore::Tile* tile = sp.store->tile(region);
    // Not resident: issueRequests queues it again when it is
    if (!tile) return;
    ensurePool();
    // A copy of the merged level-0 heights (main thread: one add per sample)
    Vulkan::TerrainTileData merged = tile->data;
    {
        VisualTimer t(visualMs);
        TerrainVisualPatch::mergeTile(*edits, region, layout(chunk.space), tile->data, merged);
    }
    auto result                           = std::make_shared<Vulkan::GrassMaskData>();
    const Vulkan::TerrainLayout lay       = layout(chunk.space);
    const std::optional<int32_t> sea      = sp.settings.seaLevel;
    const std::optional<HookGrassRule> rl = sp.grassRule;
    const bool generator                  = sp.generatorMasks;
    const ContentSource content           = sp.content;
    const uint16_t version                = generator ? placementVersion(chunk) : uint16_t(0);
    const std::function<void()> hook      = maskJobHookForTests;
    // F1: only the terrain-only revision drops a result (a tombstone bumps the sequence, never the revision)
    const uint64_t revision = edits->revision(chunk), generation = setGeneration, thresholds = maskGeneration;
    editedJobs.insert(chunk);
    jobsSubmitted++;
    pool->submit(
        namedJob("edited grass mask space " + std::to_string(chunk.space) + " chunk (" + std::to_string(chunk.x) +
                     ", " + std::to_string(chunk.z) + ")",
                 [result, merged, lay, sea, rl, generator, content, chunk, version, hook]()
                 {
                     if (hook) hook();
                     // S23b (E3): a generator space ANDs its generator mask with S20's rule over the merged heights
                     // (its grassEditRule; none: the generator mask); a legacy space S22's rule alone
                     if (generator)
                     {
                         // The owner lives until the mask is copied (the cache may evict the placement meanwhile)
                         const std::shared_ptr<const ChunkPlacement> placement = content.placementInJob(chunk, version);
                         const GrassMask& g                                    = placement->grass;
                         *result = rl ? andRule(g, generateGrassMask(merged, lay, sea, rl->minNormalY, rl->minAboveSea))
                                      : fromGenerator(g);
                     }
                     else if (rl)
                         *result = generateGrassMask(merged, lay, sea, rl->minNormalY, rl->minAboveSea);
                     else
                         *result = zeroMask(lay);
                 }),
        100000,
        [this, chunk, result, revision, generation, thresholds, generator, version](bool cancelled)
        {
            editedJobs.erase(chunk);
            // Another GPU set, other sources, or the chunk left the desired set: discarded, no re-queue
            if (cancelled || generation != setGeneration || thresholds != maskGeneration || !desiredMasks.count(chunk))
                return;
            VisualTimer t(visualMs);
            if (!edits || edits->revision(chunk) != revision || (generator && version != placementVersion(chunk)))
            {
                // The terrain (or the chunk's placement version) changed meanwhile: dropped and recomputed
                statistics.maskResultsDropped++;
                queueMask(chunk);
                return;
            }
            visual->masks[chunk] = TerrainVisualPatch::EditedMask{std::move(*result), revision, version};
            statistics.maskRecomputes++;
            // A sent mask is replaced at once; a new one is sent by sendUpdates (after its removals)
            if (sink && setActive && setSpace == chunk.space && sentMasks.count(chunk))
                sendGrassMask(chunk, visual->masks[chunk].mask);
        });
}

void TerrainSystem::terrainChanged(SpaceId space, const SampleRect& box)
{
    if (!sink || box.empty()) return;
    if (viewSpaceValid && space == viewSpace) dirty.addTileBox(box);
    dirty.addPyramidBox(space, box);
}

int64_t TerrainSystem::patchTileBox(SpaceId space, const SampleRect& box, bool patch)
{
    const Space& sp = spaces[space];
    const int64_t S = 256 / sp.settings.baseSpacing;
    const int N     = int(S) + 1;
    int64_t n       = 0;
    for (const RegionId& r : sentTiles)
    {
        if (r.space != space) continue;
        const int64_t span = int64_t(1) << r.level;
        const int64_t ti0 = int64_t(r.x) * span * S, tk0 = int64_t(r.z) * span * S;
        const int64_t ti1 = ti0 + span * S, tk1 = tk0 + span * S;
        const int64_t lo_i = std::max(box.i0, ti0), hi_i = std::min(box.i1, ti1);
        const int64_t lo_k = std::max(box.k0, tk0), hi_k = std::min(box.k1, tk1);
        if (lo_i > hi_i || lo_k > hi_k) continue;
        // The coincident samples: indices divisible by 2^L
        const int64_t a0 = (lo_i - ti0 + span - 1) / span, a1 = (hi_i - ti0) / span;
        const int64_t b0 = (lo_k - tk0 + span - 1) / span, b1 = (hi_k - tk0) / span;
        if (a0 > a1 || b0 > b1) continue;
        const int w = int(a1 - a0 + 1), h = int(b1 - b0 + 1);
        n += int64_t(w) * h;
        if (!patch) continue;
        // The base from the store's own tile (a sent tile is resident: removed from the sink before its eviction)
        const TerrainStore::Tile* tile = sp.store->tile(r);
        SAIGA_ASSERT(tile, "TerrainSystem: a sent tile is not in the store");
        patchScratch.resize(size_t(w) * size_t(h));
        TerrainVisualPatch::mergedHeights(*edits, space, ti0 + a0 * span, tk0 + b0 * span, span, w, h,
                                          tile->data.heights->data() + size_t(b0 * N + a0), size_t(N),
                                          patchScratch.data());
        sink->patchTerrainTile(r, int(a0), int(b0), w, h, patchScratch.data());
        statistics.patchCalls++;
    }
    return n;
}

int64_t TerrainSystem::patchPyramidBox(SpaceId space, const SampleRect& box, bool patch)
{
    const Space& sp                     = spaces[space];
    const Vulkan::TerrainPyramidData* p = sp.store->pyramid();
    if (!p) return 0;
    const int64_t step = Vulkan::TerrainLayout::pyramidSpacing / sp.settings.baseSpacing;
    const int P        = p->layout.pyramidSamplesPerSide();
    const int64_t a0 = (std::max<int64_t>(box.i0, 0) + step - 1) / step, a1 = std::min<int64_t>(box.i1 / step, P - 1);
    const int64_t b0 = (std::max<int64_t>(box.k0, 0) + step - 1) / step, b1 = std::min<int64_t>(box.k1 / step, P - 1);
    if (a0 > a1 || b0 > b1) return 0;
    const int w = int(a1 - a0 + 1), h = int(b1 - b0 + 1);
    if (patch)
    {
        patchScratch.resize(size_t(w) * size_t(h));
        TerrainVisualPatch::mergedHeights(*edits, space, a0 * step, b0 * step, step, w, h,
                                          p->heights->data() + size_t(b0 * P + a0), size_t(P), patchScratch.data());
        sink->patchTerrainPyramid(space, int(a0), int(b0), w, h, patchScratch.data());
        statistics.patchCalls++;
    }
    return int64_t(w) * h;
}

int64_t TerrainSystem::patchEditedTiles(SpaceId space, int64_t budget)
{
    if (!sink || !setActive || setSpace != space || !edits) return 0;
    VisualTimer timer(visualMs);
    int64_t used = 0;
    // Drains one list in insertion order; a box over the remaining budget is split by rows (the first rows now, the
    // rest stays at the front); at least one row per frame
    // The sent masks of the level-0 chunks a patched box touches are recomputed
    auto queueMasks = [&](const SampleRect& b)
    {
        const int64_t S = 256 / spaces[space].settings.baseSpacing;
        for (const ChunkId& c : sentMasks)
            if (c.space == space && int64_t(c.x) * S <= b.i1 && int64_t(c.x) * S + S >= b.i0 &&
                int64_t(c.z) * S <= b.k1 && int64_t(c.z) * S + S >= b.k0)
                queueMask(c);
    };
    auto drain = [&](TerrainDirtyQueue::List& l, bool tiles)
    {
        std::vector<SampleRect>& list = l.boxes;
        auto cost                     = [&](const SampleRect& b)
        { return tiles ? patchTileBox(space, b, false) : patchPyramidBox(space, b, false); };
        while (!list.empty())
        {
            const SampleRect box    = list.front();
            const int64_t remaining = budget < 0 ? std::numeric_limits<int64_t>::max() : budget - used;
            if (remaining <= 0) return false;
            const int64_t c = cost(box);
            if (c <= remaining)
            {
                used += tiles ? patchTileBox(space, box, true) : patchPyramidBox(space, box, true);
                if (tiles) queueMasks(box);
                list.erase(list.begin());
                l.frontActive = false;
                continue;
            }
            // The most rows that fit (cost is monotone in the rows)
            int64_t lo = 0, hi = box.height() - 1;
            while (lo < hi)
            {
                const int64_t mid = (lo + hi + 1) / 2;
                SampleRect head   = box;
                head.k1           = box.k0 + mid - 1;
                if (cost(head) <= remaining)
                    lo = mid;
                else
                    hi = mid - 1;
            }
            if (lo == 0)
            {
                if (used > 0) return false;
                lo = 1;
            }
            SampleRect head = box;
            head.k1         = box.k0 + lo - 1;
            used += tiles ? patchTileBox(space, head, true) : patchPyramidBox(space, head, true);
            if (tiles) queueMasks(head);
            list.front().k0 = box.k0 + lo;
            l.frontActive   = true;
            return false;
        }
        return true;
    };
    if (drain(dirty.tileList(), true)) drain(dirty.pyramidList(space), false);
    statistics.patchedSamples += uint64_t(used);
    statistics.patchedSamplesLastUpdate += uint64_t(used);
    return used;
}

void TerrainSystem::sendGrassMask(const ChunkId& chunk, const Vulkan::GrassMaskData& mask)
{
    // S22: a chunk with deltas gets its cached edited mask while its revision is current; else nothing yet (never the
    // base grass on a crater wall) and a recompute
    if (editedChunk(chunk))
    {
        const Vulkan::GrassMaskData* edited = visual->currentMask(chunk, edits->revision(chunk));
        if (!edited)
        {
            queueMask(chunk);
            return;
        }
        sink->setGrassMask(chunk, *edited);
        sentMasks.insert(chunk);
        return;
    }
    sink->setGrassMask(chunk, mask);
    sentMasks.insert(chunk);
}

void TerrainSystem::beginView(SpaceId space)
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown view space");
    const bool terrain = spaces[space].hasTerrain;
    if (viewSpaceValid && viewSpace != space)
    {
        // S22: the tile boxes belong to the old view space, the edited masks to its GPU set
        dirty.clearTileBoxes();
        visual->masks.clear();
        maskQueue.clear();
        baseResend.clear();
        // The old space's streamed CPU tiles and masks go (its pyramid stays cached) and its requests are cancelled
        spaces[viewSpace].store->dropStreamed();
        // With them their placement versions (eviction only visits the masks of the view space)
        for (auto it = maskVersions.begin(); it != maskVersions.end();)
            it = it->first.space == viewSpace ? maskVersions.erase(it) : std::next(it);
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
    // A new set: the sink drops the old one itself when it receives the pyramid; the tile boxes of the old set go (the
    // new set's tiles are sent merged, S22). Without a cached pyramid the old set is cleared now (else it stayed in the
    // sink, e.g. through a later switch to a flat space)
    dirty.clearTileBoxes();
    setGeneration++;
    const bool hadSet = setActive;
    setActive         = false;
    sentTiles.clear();
    sentMasks.clear();
    if (const Vulkan::TerrainPyramidData* p = spaces[space].store->pyramid())
        sendPyramid(space, *p);
    else if (hadSet)
        sink->clearTerrain();
}

void TerrainSystem::updateView(SpaceId space, const Vec3& camera)
{
    updateTerrainView(space, camera);
    // S23b: the content sets (also in a space without terrain), after the drain of this update
    if (contentView && sink && camera.allFinite()) contentView(space, camera, false, 0);
}

bool TerrainSystem::completeView(SpaceId space, const Vec3& camera, double timeoutSeconds)
{
    // One deadline for both parts: no content build starts after it (the view is then incomplete)
    const double deadline = nowSeconds() + timeoutSeconds;
    bool complete         = completeTerrainView(space, camera, timeoutSeconds);
    // S23b: the content sets are built synchronously (screenshots and goldens see them)
    if (contentView && sink && camera.allFinite()) complete = contentView(space, camera, true, deadline) && complete;
    return complete;
}

void TerrainSystem::updateTerrainView(SpaceId space, const Vec3& camera)
{
    frame++;
    visualMs                            = 0;
    statistics.patchedSamplesLastUpdate = 0;
    // Also for the early returns (sendUpdates sets it when it runs)
    statistics.uploadsLastUpdate = 0;
    drainPool();
    // Without a sink, or with a non-finite camera (nothing is desired, cancelled or evicted for it)
    if (!sink || !camera.allFinite())
    {
        updateStats(space, nullptr);
        return;
    }
    viewCamera = camera;
    beginView(space);
    if (!spaces[space].hasTerrain)
    {
        updateStats(space, nullptr);
        return;
    }
    Space& sp = spaces[space];
    if (!sp.store->pyramid())
    {
        requestPyramid(space);
        updateStats(space, nullptr);
        return;
    }
    if (!setActive) sendPyramid(space, *sp.store->pyramid());
    const Desired d = computeDesired(space, camera);
    desiredMasks    = d.maskSet;
    markDesired(space, d);
    cancelUndesired(space, d);
    issueRequests(space, d, size_t(std::max(1, view.maxInFlight)));
    evict(space, d);
    sendUpdates(space, d, std::max(0, view.uploadsPerFrame));
    // S22: the changed boxes of terrain edits, within the per-frame budget; then the edited masks, capped
    patchEditedTiles(space, std::max<int64_t>(0, view.editSamplesPerFrame));
    statistics.maskJobsLastUpdate = processMaskQueue(std::max(0, view.editMasksPerFrame));
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

bool TerrainSystem::completeTerrainView(SpaceId space, const Vec3& camera, double timeoutSeconds)
{
    const double start    = nowSeconds();
    const double deadline = start + timeoutSeconds;
    frame++;
    visualMs                            = 0;
    statistics.patchedSamplesLastUpdate = 0;
    drainPool();
    auto finish = [&](bool ok, const Desired* d)
    {
        statistics.lastCompleteMs = (nowSeconds() - start) * 1000.0;
        updateStats(space, d);
        return ok;
    };
    // A non-finite camera of a terrain space changes nothing (the view, the sink's set, the saved camera and the
    // statistics of the view space stay)
    if (spaces[space].hasTerrain && !camera.allFinite())
    {
        SAIGA_LOG("terrain", "completeView: the camera of space %u is not finite", unsigned(space));
        statistics.lastCompleteMs = (nowSeconds() - start) * 1000.0;
        return false;
    }
    if (camera.allFinite()) viewCamera = camera;
    beginView(space);
    if (!spaces[space].hasTerrain) return finish(true, nullptr);
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
    desiredMasks    = d.maskSet;
    markDesired(space, d);
    cancelUndesired(space, d);
    statistics.maskJobsLastUpdate = 0;
    int before                    = missingCounts(space, d).total();
    for (;;)
    {
        const uint64_t submitted = jobsSubmitted;
        // (b) every missing desired tile
        issueRequests(space, d, std::numeric_limits<size_t>::max());
        statistics.maskJobsLastUpdate += processMaskQueue(-1);
        runUntil(deadline);
        drainPool();
        // (c) the masks whose level-0 tile is now stored (S22: and the edited masks, waited for)
        issueRequests(space, d, std::numeric_limits<size_t>::max());
        statistics.maskJobsLastUpdate += processMaskQueue(-1);
        runUntil(deadline);
        drainPool();
        // (d) eviction and every send, then every changed box of terrain edits (S22: no budget) and their masks
        evict(space, d);
        sendUpdates(space, d, -1);
        patchEditedTiles(space, -1);
        statistics.maskJobsLastUpdate += processMaskQueue(-1);
        const MissingCounts m = missingCounts(space, d);
        if (m.total() == 0) return finish(true, &d);
        if (nowSeconds() >= deadline) return finish(false, &d);
        // Nothing arrived, no job was submitted (a dropped result is requested again) and nothing is queued or running:
        // no later iteration can progress either
        if (m.total() >= before && requests.empty() && editedJobs.empty() && maskQueue.empty() &&
            jobsSubmitted == submitted)
        {
            SAIGA_LOG("terrain",
                      "completeView: space %u cannot complete (tiles not stored %d, not sent %d; masks not stored %d, "
                      "not sent %d)",
                      unsigned(space), m.tilesNotStored, m.tilesNotSent, m.masksNotStored, m.masksNotSent);
            return finish(false, &d);
        }
        before = m.total();
    }
}

bool TerrainSystem::viewComplete(SpaceId space, const Vec3& camera) const
{
    if (!hasTerrain(space)) return true;
    if (!spaces[space].store->pyramid() || !camera.allFinite()) return false;
    if (sink && (!setActive || setSpace != space || setLayout != layout(space))) return false;
    return !missing(space, computeDesired(space, camera));
}

double TerrainSystem::readiness(SpaceId space, const Vec3& camera, double radius) const
{
    if (!hasTerrain(space)) return 1;
    // Also 0 for a stale layout (the new set is sent at the next update)
    if (!sink || !setActive || setSpace != space || setLayout != layout(space) || !camera.allFinite()) return 0;
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
    view = settings;
}

void TerrainSystem::replaceSources(std::vector<TerrainHeightSource> sources, std::vector<ContentSource> content)
{
    SAIGA_ASSERT(sources.size() == spaces.size(), "TerrainSystem::replaceSources: one source per space");
    SAIGA_ASSERT(content.empty() || content.size() == spaces.size(), "TerrainSystem::replaceSources: content sources");
    for (size_t i = 0; i < content.size(); ++i) spaces[i].content = std::move(content[i]);
    maskVersions.clear();
    // The old jobs: queued ones are cancelled, running ones joined, their completions discarded (a new pool on demand)
    if (pool && beforePoolReset) beforePoolReset();
    pool.reset();
    requests.clear();
    for (size_t i = 0; i < spaces.size(); ++i)
    {
        spaces[i].source = std::move(sources[i]);
        spaces[i].store->dropAll();
    }
    dirty.clear();
    visual->masks.clear();
    maskQueue.clear();
    editedJobs.clear();
    baseResend.clear();
    desiredMasks.clear();
    maskGeneration++;
    setGeneration++;
    if (sink && setActive) sink->clearTerrain();
    setActive = false;
    sentTiles.clear();
    sentMasks.clear();
}

void TerrainSystem::pinSimulationTile(const ChunkId& chunk, const std::shared_ptr<const ChunkBase>& base)
{
    SAIGA_ASSERT(chunk.space < spaces.size() && base && base->heights, "TerrainSystem::pinSimulationTile: no base");
    Vulkan::TerrainTileData t;
    t.heights = base->heights;
    Vulkan::terrainHeightRange(*t.heights, t.minHeight, t.maxHeight);
    spaces[chunk.space].store->pinTile(regionOf(chunk.space, 0, chunk.x, chunk.z), std::move(t));
}

void TerrainSystem::unpinSimulationTile(const ChunkId& chunk)
{
    if (chunk.space >= spaces.size()) return;
    spaces[chunk.space].store->unpinTile(regionOf(chunk.space, 0, chunk.x, chunk.z));
}

void TerrainSystem::setSink(RenderSink* s)
{
    // Regression (CodeRabbit, S20): a sink attached again kept the tiles of its old set (the store may hold others
    // now); a pyramid of the same space and layout keeps them, and the replay overflowed tilesPerLevel
    if (sink && setActive && std::find(staleSinks.begin(), staleSinks.end(), sink) == staleSinks.end())
        staleSinks.push_back(sink);
    if (s)
    {
        auto it = std::find(staleSinks.begin(), staleSinks.end(), s);
        if (it != staleSinks.end())
        {
            staleSinks.erase(it);
            s->clearTerrain();
        }
    }
    sink      = s;
    setActive = false;
    // S22: a new sink receives merged data; without a sink nothing is queued
    dirty.clear();
    maskQueue.clear();
    baseResend.clear();
    setGeneration++;
    sentTiles.clear();
    sentMasks.clear();
    if (!sink || !viewSpaceValid || !spaces[viewSpace].hasTerrain) return;
    const TerrainStore& st = *spaces[viewSpace].store;
    if (!st.pyramid()) return;
    // What the store holds of the last view camera's desired set (truncated to the current budget: the store can hold
    // more after setViewSettings lowered it), through the send point
    sendPyramid(viewSpace, *st.pyramid());
    const Desired d = computeDesired(viewSpace, viewCamera);
    for (int k = 0; k < 8; ++k)
        for (const auto& t : d.tiles[size_t(k)])
            if (const TerrainStore::Tile* tile = st.tile(t.second)) sendTile(t.second, tile->data);
    for (const auto& m : d.masks)
    {
        if (const Vulkan::GrassMaskData* mask = st.mask(m.second))
            sendGrassMask(m.second, *mask);
        else if (auto it = visual->masks.find(m.second); it != visual->masks.end() && editedChunk(m.second))
            // The edited mask of a chunk without a base mask (S22), only while the chunk has deltas (PR review: a mask
            // of deltas returned to zero went out as an unedited chunk's mask; sendGrassMask checks its revision)
            sendGrassMask(m.second, it->second.mask);
    }
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
    s.masksResident    = int(st.masks().size());
    s.masksSent        = int(sentMasks.size());
    s.pending          = int(requests.size());
    s.viewSpace        = space;
    s.queuedBoxes      = dirty.size();
    s.lastVisualMs     = visualMs;
    s.maskQueue        = int(maskQueue.size());
    s.maskJobsInFlight = int(editedJobs.size());
    size_t bytes       = 0;
    s.pinnedBytes      = 0;
    for (const Space& sp : spaces)
    {
        bytes += sp.store->cpuBytes();
        s.pinnedBytes += sp.store->pinnedBytes();
    }
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
