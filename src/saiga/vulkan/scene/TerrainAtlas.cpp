/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/TerrainAtlas.h"

#include "saiga/core/util/assert.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
size_t align16(size_t v)
{
    return (v + 15) & ~size_t(15);
}

int nodesPerSideOf(const TerrainLayout& l, int level)
{
    return int((int64_t(l.sizeChunks) * 8 / l.baseSpacing) >> level);
}

size_t totalNodes(const TerrainLayout& l)
{
    size_t n = 0;
    for (int k = 0; k <= l.rootLevel(); ++k) n += size_t(nodesPerSideOf(l, k)) * size_t(nodesPerSideOf(l, k));
    return n;
}

size_t pageEntries(const TerrainLayout& l)
{
    size_t n = 0;
    for (int k = 0; k < l.streamedLevels(); ++k) n += size_t(l.tilesPerSide(k)) * size_t(l.tilesPerSide(k));
    return n;
}

// A set buffer of 0 bytes (e.g. no grass layers) is allocated with this size
size_t bufferBytes(size_t bytes)
{
    return std::max<size_t>(bytes, 16);
}

void transferBarrier(vk::CommandBuffer cmd)
{
    vk::MemoryBarrier2 b(Stage::eAllTransfer, Access::eTransferWrite, Stage::eAllTransfer,
                         Access::eTransferWrite | Access::eTransferRead);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, b, nullptr, nullptr));
}
}  // namespace

TerrainMemory TerrainAtlas::memoryEstimate(const TerrainLayout& l, const RenderSceneParameters& p)
{
    TerrainMemory m;
    const size_t n = size_t(l.samplesPerSide());
    const size_t P = size_t(l.pyramidSamplesPerSide());
    m.atlas        = bufferBytes(size_t(l.streamedLevels()) * size_t(l.tilesPerLevel) * n * n * 4);
    m.pyramid      = bufferBytes(P * P * 4);
    m.pageTable    = bufferBytes(pageEntries(l) * 4);
    m.masks        = bufferBytes(size_t(l.grassMaskLayers) * size_t(l.grassMaskSize()) * size_t(l.grassMaskSize()));
    m.nodes        = 2 * size_t(maxNodes) * nodeInstanceBytes;
    m.staging      = 2 * (p.terrainUploadBytesPerFrame + stagingMargin);
    m.bounds       = totalNodes(l) * sizeof(NodeBounds);
    m.uploadData   = p.terrainUploadBytesPerFrame;
    return m;
}

void TerrainAtlas::init(VulkanBase& b, const RenderSceneParameters& p, uint64_t limit)
{
    destroy();
    base         = &b;
    params       = p;
    storageLimit = limit;
    SAIGA_ASSERT(params.terrainUploadBytesPerFrame >= 2 * (size_t(257) * 257 * 4 + 32),
                 "RenderSceneParameters::terrainUploadBytesPerFrame must hold at least two 1 m tiles");
    for (int f = 0; f < 2; ++f)
    {
        staging[size_t(f)].createBuffer(
            *base, params.terrainUploadBytesPerFrame + stagingMargin, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        nodeBuffers[size_t(f)].createBuffer(
            *base, size_t(maxNodes) * nodeInstanceBytes, vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
    initialized = true;
    statistics  = Stats();
    updateMemoryStats();
}

void TerrainAtlas::destroy()
{
    if (!initialized) return;
    dropSet();
    for (auto& s : staging) s.destroy();
    for (auto& n : nodeBuffers) n.destroy();
    initialized = false;
    base        = nullptr;
}

Buffer& TerrainAtlas::createSetBuffer(Buffer& b, size_t bytes, const char* what)
{
    bytes = bufferBytes(bytes);
    if (bytes > storageLimit)
        SAIGA_EXIT_ERROR("RenderScene: the terrain " + std::string(what) + " of this layout needs " +
                         std::to_string(bytes) + " bytes, more than the storage buffer limit (maxStorageBufferRange " +
                         std::to_string(storageLimit) + " bytes)");
    b.createBuffer(*base, bytes,
                   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                       vk::BufferUsageFlagBits::eTransferSrc,
                   vk::MemoryPropertyFlagBits::eDeviceLocal);
    return b;
}

void TerrainAtlas::allocateSet(SpaceId space, const TerrainPyramidData& p)
{
    dropSet();
    const TerrainLayout& l = p.layout;
    SAIGA_ASSERT(l.baseSpacing == 1 || l.baseSpacing == 2, "TerrainLayout: baseSpacing 1 or 2");
    SAIGA_ASSERT(l.sizeChunks >= 16 && l.sizeChunks % 16 == 0, "TerrainLayout: sizeChunks a multiple of 16");
    SAIGA_ASSERT(l.tilesPerLevel >= 1 && l.grassMaskLayers >= 0, "TerrainLayout: the budgets");
    setLayout = l;
    setSpace  = space;
    // Checked first (fatal), then allocated: the old buffers are released after the frames in flight
    const TerrainMemory m = memoryEstimate(l, params);
    createSetBuffer(atlas, m.atlas, "atlas");
    createSetBuffer(pyramid, m.pyramid, "pyramid");
    createSetBuffer(pages, m.pageTable, "page table");
    createSetBuffer(masks, m.masks, "grass masks");
    pageOffsets.assign(size_t(l.streamedLevels()), 0);
    uint32_t offset = 0;
    tileSlots.assign(size_t(l.streamedLevels()), {});
    tiles.assign(size_t(l.streamedLevels()), {});
    for (int k = 0; k < l.streamedLevels(); ++k)
    {
        pageOffsets[size_t(k)] = offset;
        const size_t entries   = size_t(l.tilesPerSide(k)) * size_t(l.tilesPerSide(k));
        offset += uint32_t(entries);
        tileSlots[size_t(k)].assign(entries, -1);
        tiles[size_t(k)].assign(size_t(l.tilesPerLevel), TileRecord());
    }
    maskRecords.assign(size_t(l.grassMaskLayers), MaskRecord());
    maskByChunk.clear();
    nodes.assign(size_t(l.rootLevel()) + 1, {});
    for (int k = 0; k <= l.rootLevel(); ++k)
        nodes[size_t(k)].assign(size_t(nodesPerSide(k)) * size_t(nodesPerSide(k)), NodeBounds());
    pageClearPending = true;
    setAllocated     = true;
    statistics.rebuilds++;
}

void TerrainAtlas::dropSet()
{
    atlas.destroy();
    pyramid.destroy();
    pages.destroy();
    masks.destroy();
    queue.clear();
    queuedBytes = 0;
    tiles.clear();
    tileSlots.clear();
    maskRecords.clear();
    maskByChunk.clear();
    nodes.clear();
    pageOffsets.clear();
    pyramidHeights.reset();
    setSeaLevel.reset();
    pyramidGpu       = false;
    pageClearPending = false;
    setAllocated     = false;
}

void TerrainAtlas::clear()
{
    dropSet();
    updateMemoryStats();
}

void TerrainAtlas::enqueue(Upload u)
{
    queuedBytes += u.bytes;
    queue.push_back(std::move(u));
}

void TerrainAtlas::dropQueuedTile(const RegionId& region)
{
    for (auto it = queue.begin(); it != queue.end();)
    {
        if ((it->kind == Kind::Tile || it->kind == Kind::Patch) && it->region == region)
        {
            queuedBytes -= it->bytes;
            it = queue.erase(it);
        }
        else
            ++it;
    }
}

void TerrainAtlas::queuePyramidRows()
{
    const int P           = setLayout.pyramidSamplesPerSide();
    const size_t rowBytes = size_t(P) * 4;
    const int perItem     = int(std::max<size_t>(1, (params.terrainUploadBytesPerFrame - 16) / rowBytes));
    for (int r = 0; r < P; r += perItem)
    {
        Upload u;
        u.kind    = Kind::PyramidRows;
        u.heights = pyramidHeights;
        u.row0    = r;
        u.rows    = std::min(perItem, P - r);
        u.bytes   = align16(size_t(u.rows) * rowBytes);
        enqueue(std::move(u));
    }
}

void TerrainAtlas::setPyramid(SpaceId space, const TerrainPyramidData& p)
{
    SAIGA_ASSERT(initialized);
    const int P = p.layout.pyramidSamplesPerSide();
    SAIGA_ASSERT(p.heights && p.heights->size() == size_t(P) * size_t(P), "setTerrainPyramid: wrong sample count");
    if (!setAllocated || space != setSpace || p.layout != setLayout)
    {
        allocateSet(space, p);
    }
    else
    {
        // The same space and layout: the pyramid data is replaced in place (its queued rows and patches are dropped)
        for (auto it = queue.begin(); it != queue.end();)
        {
            if (it->kind == Kind::PyramidRows || it->kind == Kind::PyramidPatch)
            {
                queuedBytes -= it->bytes;
                it = queue.erase(it);
            }
            else
                ++it;
        }
    }
    setSeaLevel    = p.seaLevel;
    pyramidHeights = p.heights;
    queuePyramidRows();
    pyramidOwnBounds();
    updateMemoryStats();
}

TerrainAtlas::TileRecord* TerrainAtlas::findTile(const RegionId& region)
{
    if (!setAllocated || region.space != setSpace || region.level >= setLayout.streamedLevels()) return nullptr;
    const int32_t W = setLayout.tilesPerSide(region.level);
    if (region.x < 0 || region.z < 0 || region.x >= W || region.z >= W) return nullptr;
    const int slot = tileSlots[region.level][size_t(region.z) * size_t(W) + size_t(region.x)];
    return slot < 0 ? nullptr : &tiles[region.level][size_t(slot)];
}

bool TerrainAtlas::tileSet(int level, int32_t tx, int32_t tz) const
{
    if (!setAllocated || level >= setLayout.streamedLevels()) return false;
    const int32_t W = setLayout.tilesPerSide(level);
    if (tx < 0 || tz < 0 || tx >= W || tz >= W) return false;
    return tileSlots[size_t(level)][size_t(tz) * size_t(W) + size_t(tx)] >= 0;
}

bool TerrainAtlas::tileResident(int level, int32_t tx, int32_t tz) const
{
    if (!tileSet(level, tx, tz)) return false;
    const int32_t W = setLayout.tilesPerSide(level);
    const int slot  = tileSlots[size_t(level)][size_t(tz) * size_t(W) + size_t(tx)];
    return tiles[size_t(level)][size_t(slot)].gpu;
}

void TerrainAtlas::setTile(const RegionId& region, const TerrainTileData& tile)
{
    SAIGA_ASSERT(setAllocated && region.space == setSpace,
                 "setTerrainTile: no terrain set of this space (pyramid first)");
    SAIGA_ASSERT(region.level < setLayout.streamedLevels(), "setTerrainTile: not a streamed level");
    const int32_t W = setLayout.tilesPerSide(region.level);
    SAIGA_ASSERT(region.x >= 0 && region.z >= 0 && region.x < W && region.z < W, "setTerrainTile: outside the space");
    const size_t n = size_t(setLayout.samplesPerSide());
    SAIGA_ASSERT(tile.heights && tile.heights->size() == n * n, "setTerrainTile: wrong sample count");
    int& slot  = tileSlots[region.level][size_t(region.z) * size_t(W) + size_t(region.x)];
    auto& recs = tiles[region.level];
    if (slot >= 0)
    {
        // Replaced in place: the queued copy and patches of the old data are superseded
        dropQueuedTile(region);
    }
    else
    {
        int free = -1;
        for (size_t i = 0; i < recs.size(); ++i)
            if (!recs[i].used)
            {
                free = int(i);
                break;
            }
        SAIGA_ASSERT(free >= 0, "setTerrainTile: no free layer on this level (tilesPerLevel)");
        slot                    = free;
        recs[size_t(free)]      = TileRecord();
        recs[size_t(free)].used = true;
    }
    TileRecord& r = recs[size_t(slot)];
    r.region      = region;
    r.data        = tile;
    Upload u;
    u.kind      = Kind::Tile;
    u.region    = region;
    u.layer     = region.level * setLayout.tilesPerLevel + slot;
    u.heights   = tile.heights;
    u.pageIndex = pageOffsets[region.level] + uint32_t(region.z) * uint32_t(W) + uint32_t(region.x);
    u.pageValue = u.layer;
    u.bytes     = align16(layerBytes()) + 16;
    enqueue(std::move(u));
    tileOwnBounds(region.level, region.x, region.z, &r.data);
    updateMemoryStats();
}

void TerrainAtlas::removeTile(const RegionId& region)
{
    TileRecord* r = findTile(region);
    if (!r) return;
    const int32_t W = setLayout.tilesPerSide(region.level);
    int& slot       = tileSlots[region.level][size_t(region.z) * size_t(W) + size_t(region.x)];
    dropQueuedTile(region);
    Upload u;
    u.kind      = Kind::Page;
    u.region    = region;
    u.pageIndex = pageOffsets[region.level] + uint32_t(region.z) * uint32_t(W) + uint32_t(region.x);
    u.pageValue = -1;
    u.bytes     = 16;
    enqueue(std::move(u));
    *r   = TileRecord();
    slot = -1;
    tileOwnBounds(region.level, region.x, region.z, nullptr);
    updateMemoryStats();
}

void TerrainAtlas::setGrassMask(const ChunkId& chunk, const GrassMaskData& mask)
{
    SAIGA_ASSERT(setAllocated && chunk.space == setSpace, "setGrassMask: no terrain set of this space");
    SAIGA_ASSERT(mask.density && mask.size == setLayout.grassMaskSize() &&
                     mask.density->size() == size_t(mask.size) * size_t(mask.size),
                 "setGrassMask: wrong mask size");
    const auto key = std::make_pair(chunk.x, chunk.z);
    auto it        = maskByChunk.find(key);
    int layer      = -1;
    if (it != maskByChunk.end())
    {
        layer = it->second;
        for (auto q = queue.begin(); q != queue.end();)
        {
            if (q->kind == Kind::Mask && q->layer == layer)
            {
                queuedBytes -= q->bytes;
                q = queue.erase(q);
            }
            else
                ++q;
        }
    }
    else
    {
        for (size_t i = 0; i < maskRecords.size(); ++i)
            if (!maskRecords[i].used)
            {
                layer = int(i);
                break;
            }
        SAIGA_ASSERT(layer >= 0, "setGrassMask: no free mask layer (grassMaskLayers)");
        maskRecords[size_t(layer)] = MaskRecord{chunk.x, chunk.z, true, false};
        maskByChunk[key]           = layer;
    }
    Upload u;
    u.kind  = Kind::Mask;
    u.cx    = chunk.x;
    u.cz    = chunk.z;
    u.layer = layer;
    u.mask  = mask.density;
    u.bytes = align16(maskLayerBytes());
    enqueue(std::move(u));
    updateMemoryStats();
}

void TerrainAtlas::removeGrassMask(const ChunkId& chunk)
{
    if (!setAllocated || chunk.space != setSpace) return;
    auto it = maskByChunk.find(std::make_pair(chunk.x, chunk.z));
    if (it == maskByChunk.end()) return;
    const int layer = it->second;
    for (auto q = queue.begin(); q != queue.end();)
    {
        if (q->kind == Kind::Mask && q->layer == layer)
        {
            queuedBytes -= q->bytes;
            q = queue.erase(q);
        }
        else
            ++q;
    }
    maskRecords[size_t(layer)] = MaskRecord();
    maskByChunk.erase(it);
    updateMemoryStats();
}

int TerrainAtlas::maskLayer(int32_t cx, int32_t cz) const
{
    if (!setAllocated) return -1;
    auto it = maskByChunk.find(std::make_pair(cx, cz));
    if (it == maskByChunk.end()) return -1;
    return maskRecords[size_t(it->second)].gpu ? it->second : -1;
}

void TerrainAtlas::patchTile(const RegionId& region, int x0, int z0, int w, int h, const int32_t* heights)
{
    TileRecord* r = findTile(region);
    if (!r)
    {
        statistics.patchesIgnored++;
        return;
    }
    const int n = setLayout.samplesPerSide();
    SAIGA_ASSERT(w > 0 && h > 0 && x0 >= 0 && z0 >= 0 && x0 + w <= n && z0 + h <= n,
                 "patchTerrainTile: the rectangle must lie inside the tile");
    const int32_t W    = setLayout.tilesPerSide(region.level);
    const int slot     = tileSlots[region.level][size_t(region.z) * size_t(W) + size_t(region.x)];
    const size_t rowFp = align16(size_t(w) * 4);
    const int perItem  = int(std::max<size_t>(1, (params.terrainUploadBytesPerFrame - 16) / rowFp));
    int32_t lo = std::numeric_limits<int32_t>::max(), hi = std::numeric_limits<int32_t>::min();
    for (int row = 0; row < h; row += perItem)
    {
        Upload u;
        u.kind   = Kind::Patch;
        u.region = region;
        u.layer  = region.level * setLayout.tilesPerLevel + slot;
        u.x0     = x0;
        u.z0     = z0 + row;
        u.w      = w;
        u.h      = std::min(perItem, h - row);
        u.patch.assign(heights + size_t(row) * size_t(w), heights + size_t(row + u.h) * size_t(w));
        for (int32_t v : u.patch)
        {
            SAIGA_ASSERT(v >= -kTerrainHeightLimit && v <= kTerrainHeightLimit,
                         "patchTerrainTile: height beyond the limit");
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        u.bytes = rowFp * size_t(u.h);
        enqueue(std::move(u));
    }
    statistics.patchesApplied++;
    // The nodes of the tile whose samples overlap the rectangle widen
    const int npt   = 8 / setLayout.baseSpacing;
    const int32_t a = std::max(0, (x0 - 1) / 32), b = std::max(0, (z0 - 1) / 32);
    const int32_t c = std::min(npt - 1, (x0 + w - 1) / 32), d = std::min(npt - 1, (z0 + h - 1) / 32);
    widenOwn(region.level, region.x * npt + a, region.z * npt + b, region.x * npt + c, region.z * npt + d, lo, hi);
    updateMemoryStats();
}

void TerrainAtlas::patchPyramid(SpaceId space, int x0, int z0, int w, int h, const int32_t* heights)
{
    if (!setAllocated || space != setSpace)
    {
        statistics.patchesIgnored++;
        return;
    }
    const int P = setLayout.pyramidSamplesPerSide();
    SAIGA_ASSERT(w > 0 && h > 0 && x0 >= 0 && z0 >= 0 && x0 + w <= P && z0 + h <= P,
                 "patchTerrainPyramid: the rectangle must lie inside the pyramid");
    const size_t rowFp = align16(size_t(w) * 4);
    const int perItem  = int(std::max<size_t>(1, (params.terrainUploadBytesPerFrame - 16) / rowFp));
    int32_t lo = std::numeric_limits<int32_t>::max(), hi = std::numeric_limits<int32_t>::min();
    for (int row = 0; row < h; row += perItem)
    {
        Upload u;
        u.kind = Kind::PyramidPatch;
        u.x0   = x0;
        u.z0   = z0 + row;
        u.w    = w;
        u.h    = std::min(perItem, h - row);
        u.patch.assign(heights + size_t(row) * size_t(w), heights + size_t(row + u.h) * size_t(w));
        for (int32_t v : u.patch)
        {
            SAIGA_ASSERT(v >= -kTerrainHeightLimit && v <= kTerrainHeightLimit,
                         "patchTerrainPyramid: height beyond the limit");
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        u.bytes = rowFp * size_t(u.h);
        enqueue(std::move(u));
    }
    statistics.patchesApplied++;
    // The pyramid levels' nodes that draw an overlapping sample widen (node k covers samples [n * 32 st, +32 st])
    for (int k = setLayout.streamedLevels(); k <= setLayout.rootLevel(); ++k)
    {
        const int32_t span = 32 << (k - setLayout.streamedLevels());
        const int32_t last = nodesPerSide(k) - 1;
        const int32_t a = std::max(0, (x0 - 1) / span), b = std::max(0, (z0 - 1) / span);
        const int32_t c = std::min(last, (x0 + w - 1) / span), d = std::min(last, (z0 + h - 1) / span);
        widenOwn(k, a, b, c, d, lo, hi);
    }
    updateMemoryStats();
}

int TerrainAtlas::nodesPerSide(int level) const
{
    return nodesPerSideOf(setLayout, level);
}

TerrainAtlas::Bound TerrainAtlas::ownBound(int level, int32_t nx, int32_t nz) const
{
    const NodeBounds& n = nodes[size_t(level)][size_t(nz) * size_t(nodesPerSide(level)) + size_t(nx)];
    return Bound{n.ownLo, n.ownHi};
}

TerrainAtlas::Bound TerrainAtlas::bound(int level, int32_t nx, int32_t nz) const
{
    const NodeBounds& n = nodes[size_t(level)][size_t(nz) * size_t(nodesPerSide(level)) + size_t(nx)];
    return Bound{n.lo, n.hi};
}

void TerrainAtlas::tileOwnBounds(int level, int32_t tx, int32_t tz, const TerrainTileData* data)
{
    const int npt = 8 / setLayout.baseSpacing;
    const int n   = setLayout.samplesPerSide();
    for (int lz = 0; lz < npt; ++lz)
    {
        for (int lx = 0; lx < npt; ++lx)
        {
            NodeBounds& nb = node(level, tx * npt + lx, tz * npt + lz);
            nb.ownLo       = 1;
            nb.ownHi       = 0;
            if (!data) continue;
            int32_t lo = std::numeric_limits<int32_t>::max(), hi = std::numeric_limits<int32_t>::min();
            const std::vector<int32_t>& hs = *data->heights;
            for (int j = lz * 32; j <= lz * 32 + 32; ++j)
                for (int i = lx * 32; i <= lx * 32 + 32; ++i)
                {
                    const int32_t v = hs[size_t(j) * size_t(n) + size_t(i)];
                    lo              = std::min(lo, v);
                    hi              = std::max(hi, v);
                }
            nb.ownLo = lo;
            nb.ownHi = hi;
        }
    }
    updateBoundsUp(level, tx * npt, tz * npt, tx * npt + npt - 1, tz * npt + npt - 1);
}

void TerrainAtlas::pyramidOwnBounds()
{
    const int P                    = setLayout.pyramidSamplesPerSide();
    const std::vector<int32_t>& hs = *pyramidHeights;
    for (int k = setLayout.streamedLevels(); k <= setLayout.rootLevel(); ++k)
    {
        const int st = 1 << (k - setLayout.streamedLevels());
        const int N  = nodesPerSide(k);
        for (int nz = 0; nz < N; ++nz)
        {
            for (int nx = 0; nx < N; ++nx)
            {
                int32_t lo = std::numeric_limits<int32_t>::max(), hi = std::numeric_limits<int32_t>::min();
                for (int j = 0; j <= 32; ++j)
                {
                    const size_t row = size_t(nz * 32 * st + j * st) * size_t(P);
                    for (int i = 0; i <= 32; ++i)
                    {
                        const int32_t v = hs[row + size_t(nx * 32 * st + i * st)];
                        lo              = std::min(lo, v);
                        hi              = std::max(hi, v);
                    }
                }
                NodeBounds& nb = node(k, nx, nz);
                nb.ownLo       = lo;
                nb.ownHi       = hi;
            }
        }
    }
    // Every level's bounds, children first (the rectangle of all level-0 nodes halves per level)
    updateBoundsUp(0, 0, 0, nodesPerSide(0) - 1, nodesPerSide(0) - 1);
}

void TerrainAtlas::widenOwn(int level, int32_t nx0, int32_t nz0, int32_t nx1, int32_t nz1, int32_t lo, int32_t hi)
{
    for (int32_t z = nz0; z <= nz1; ++z)
        for (int32_t x = nx0; x <= nx1; ++x)
        {
            NodeBounds& nb = node(level, x, z);
            // A node without its own data (no tile) stays invalid: it is not drawn at this level
            if (nb.ownLo > nb.ownHi) continue;
            nb.ownLo = std::min(nb.ownLo, lo);
            nb.ownHi = std::max(nb.ownHi, hi);
        }
    updateBoundsUp(level, nx0, nz0, nx1, nz1);
}

void TerrainAtlas::updateBoundsUp(int level, int32_t nx0, int32_t nz0, int32_t nx1, int32_t nz1)
{
    for (int k = level; k <= setLayout.rootLevel(); ++k)
    {
        for (int32_t z = nz0; z <= nz1; ++z)
        {
            for (int32_t x = nx0; x <= nx1; ++x)
            {
                NodeBounds& nb = node(k, x, z);
                nb.lo          = nb.ownLo;
                nb.hi          = nb.ownHi;
                if (nb.lo > nb.hi || k == 0) continue;
                for (int c = 0; c < 4; ++c)
                {
                    const NodeBounds& ch = node(k - 1, 2 * x + (c & 1), 2 * z + (c >> 1));
                    if (ch.lo > ch.hi) continue;
                    nb.lo = std::min(nb.lo, ch.lo);
                    nb.hi = std::max(nb.hi, ch.hi);
                }
            }
        }
        nx0 /= 2;
        nz0 /= 2;
        nx1 /= 2;
        nz1 /= 2;
    }
}

void TerrainAtlas::recordUploads(vk::CommandBuffer cmd, uint32_t frameIndex)
{
    statistics.bytesRecorded = 0;
    if (!setAllocated)
    {
        updateMemoryStats();
        return;
    }
    if (pageClearPending)
    {
        Memory::SafeAccessor acc(*pages.memoryLocation());
        cmd.fillBuffer(pages.memoryLocation()->data, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
        pageClearPending = false;
        transferBarrier(cmd);
    }
    Buffer& stage       = staging[frameIndex];
    const size_t cap    = params.terrainUploadBytesPerFrame;
    const size_t layerB = layerBytes();
    const size_t n      = size_t(setLayout.samplesPerSide());
    const size_t P      = size_t(setLayout.pyramidSamplesPerSide());
    size_t used         = 0;
    size_t offset       = 0;
    std::vector<int> layers, maskLayers;
    std::vector<uint32_t> pageEntries;
    bool pyramidTouched = false, pyramidPatched = false;
    auto contains = [](const auto& v, auto x) { return std::find(v.begin(), v.end(), x) != v.end(); };
    auto flush    = [&]()
    {
        offset = batch.recordAt(cmd, stage, offset);
        transferBarrier(cmd);
        layers.clear();
        maskLayers.clear();
        pageEntries.clear();
        pyramidTouched = pyramidPatched = false;
        used += 16;
    };
    while (!queue.empty())
    {
        Upload& u = queue.front();
        if (used + u.bytes > cap) break;
        // Copies of one vkCmdCopyBuffer must not overlap: a later copy into a range written in this group (a patch
        // behind its tile, a page entry set and reset) starts a new group behind a barrier
        bool overlap = false;
        switch (u.kind)
        {
            case Kind::Tile:
                overlap = contains(layers, u.layer) || contains(pageEntries, u.pageIndex);
                break;
            case Kind::Page:
                overlap = contains(pageEntries, u.pageIndex);
                break;
            case Kind::Patch:
                overlap = contains(layers, u.layer);
                break;
            case Kind::PyramidRows:
                overlap = pyramidPatched;
                break;
            case Kind::PyramidPatch:
                overlap = pyramidTouched;
                break;
            case Kind::Mask:
                overlap = contains(maskLayers, u.layer);
                break;
        }
        if (overlap)
        {
            flush();
            if (used + u.bytes > cap) break;
        }
        switch (u.kind)
        {
            case Kind::Tile:
            {
                batch.add(atlas, vk::DeviceSize(u.layer) * layerB, u.heights->data(), layerB);
                batch.add(pages, vk::DeviceSize(u.pageIndex) * 4, &u.pageValue, 4);
                layers.push_back(u.layer);
                pageEntries.push_back(u.pageIndex);
                // GPU-resident from this frame (the record is current: replaced or removed tiles drop their uploads)
                const int32_t W = setLayout.tilesPerSide(u.region.level);
                const int slot  = tileSlots[u.region.level][size_t(u.region.z) * size_t(W) + size_t(u.region.x)];
                if (slot >= 0) tiles[u.region.level][size_t(slot)].gpu = true;
                break;
            }
            case Kind::Page:
                batch.add(pages, vk::DeviceSize(u.pageIndex) * 4, &u.pageValue, 4);
                pageEntries.push_back(u.pageIndex);
                break;
            case Kind::Patch:
                for (int r = 0; r < u.h; ++r)
                    batch.add(atlas, vk::DeviceSize(u.layer) * layerB + (size_t(u.z0 + r) * n + size_t(u.x0)) * 4,
                              u.patch.data() + size_t(r) * size_t(u.w), size_t(u.w) * 4);
                layers.push_back(u.layer);
                break;
            case Kind::PyramidRows:
                batch.add(pyramid, vk::DeviceSize(u.row0) * P * 4, u.heights->data() + size_t(u.row0) * P,
                          size_t(u.rows) * P * 4);
                pyramidTouched = true;
                if (size_t(u.row0 + u.rows) == P) pyramidGpu = true;
                break;
            case Kind::PyramidPatch:
                for (int r = 0; r < u.h; ++r)
                    batch.add(pyramid, (size_t(u.z0 + r) * P + size_t(u.x0)) * 4,
                              u.patch.data() + size_t(r) * size_t(u.w), size_t(u.w) * 4);
                pyramidTouched = pyramidPatched = true;
                break;
            case Kind::Mask:
                batch.add(masks, vk::DeviceSize(u.layer) * maskLayerBytes(), u.mask->data(), maskLayerBytes());
                maskLayers.push_back(u.layer);
                maskRecords[size_t(u.layer)].gpu = true;
                break;
        }
        used += u.bytes;
        queuedBytes -= u.bytes;
        queue.pop_front();
    }
    batch.recordAt(cmd, stage, offset);
    statistics.bytesRecorded = used;
    updateMemoryStats();
}

std::vector<int32_t> TerrainAtlas::download(Buffer& b, size_t offset, size_t bytes)
{
    base->device.waitIdle();
    Buffer host;
    host.createBuffer(*base, bytes, vk::BufferUsageFlagBits::eTransferDst,
                      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto& q               = base->mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    {
        Memory::SafeAccessor a(*b.memoryLocation());
        Memory::SafeAccessor c(*host.memoryLocation());
        cmd.copyBuffer(b.memoryLocation()->data, host.memoryLocation()->data, vk::BufferCopy(offset, 0, bytes));
    }
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    std::vector<int32_t> out(bytes / 4);
    std::memcpy(out.data(), host.getMappedPointer(), bytes);
    host.destroyImmediately();
    return out;
}

std::vector<int32_t> TerrainAtlas::readLayerForTests(const RegionId& region)
{
    TileRecord* r = findTile(region);
    SAIGA_ASSERT(r, "readLayerForTests: the tile is not set");
    const int32_t W = setLayout.tilesPerSide(region.level);
    const int slot  = tileSlots[region.level][size_t(region.z) * size_t(W) + size_t(region.x)];
    const int layer = region.level * setLayout.tilesPerLevel + slot;
    return download(atlas, size_t(layer) * layerBytes(), layerBytes());
}

std::vector<int32_t> TerrainAtlas::readPyramidForTests()
{
    SAIGA_ASSERT(setAllocated, "readPyramidForTests: no terrain set");
    const size_t P = size_t(setLayout.pyramidSamplesPerSide());
    return download(pyramid, 0, P * P * 4);
}

void TerrainAtlas::updateMemoryStats()
{
    Stats& s          = statistics;
    s.hasSet          = setAllocated;
    s.space           = setSpace;
    s.layout          = setLayout;
    s.pyramidResident = pyramidGpu;
    s.bytesQueued     = queuedBytes;
    s.tilesSet.fill(0);
    s.tilesResident.fill(0);
    s.masksSet = s.masksResident = 0;
    for (size_t k = 0; k < tiles.size(); ++k)
        for (const TileRecord& r : tiles[k])
        {
            s.tilesSet[k] += r.used ? 1 : 0;
            s.tilesResident[k] += r.used && r.gpu ? 1 : 0;
        }
    for (const MaskRecord& m : maskRecords)
    {
        s.masksSet += m.used ? 1 : 0;
        s.masksResident += m.used && m.gpu ? 1 : 0;
    }
    TerrainMemory m;
    m.atlas          = size_t(atlas.size());
    m.pyramid        = size_t(pyramid.size());
    m.pageTable      = size_t(pages.size());
    m.masks          = size_t(masks.size());
    m.nodes          = size_t(nodeBuffers[0].size() + nodeBuffers[1].size());
    m.staging        = size_t(staging[0].size() + staging[1].size());
    size_t nodeCount = 0;
    for (const auto& level : nodes) nodeCount += level.size();
    m.bounds     = nodeCount * sizeof(NodeBounds);
    m.uploadData = setAllocated ? params.terrainUploadBytesPerFrame : 0;
    s.memory     = m;
}

}  // namespace Vulkan
}  // namespace Saiga
