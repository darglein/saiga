/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"
#include "saiga/vulkan/scene/UploadBatch.h"

#include <array>
#include <deque>
#include <map>
#include <optional>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
// The memory of a GPU terrain set (TerrainAtlas::memoryEstimate, the allocated set: TerrainAtlas::Stats)
struct TerrainMemory
{
    // GPU: atlas, pyramid, page table, grass masks, the node instance buffers, the staging buffers
    size_t atlas = 0, pyramid = 0, pageTable = 0, masks = 0, nodes = 0, staging = 0;
    // CPU (RenderScene's side): the node bounds, the upload batch's data vector (at most the cap)
    size_t bounds = 0, uploadData = 0;
    size_t gpuBytes() const { return atlas + pyramid + pageTable + masks + nodes + staging; }
    size_t cpuBytes() const { return bounds + uploadData; }
};

/**
 * The GPU terrain set of RenderScene (S20, docs/engine_design.md §10.1, §12.2): the terrain of one space (the space
 * of the last pyramid) in int32 storage buffers, filled by buffer copies:
 * - atlas: streamedLevels * tilesPerLevel layers of samplesPerSide^2 heights (the integer heights themselves; the
 *   shader subtracts one frame-global reference height, every difference is below 2^24: exact in float);
 * - pyramid: the whole space at 32 m, (8 * sizeChunks + 1)^2 heights;
 * - page table: per streamed level a grid of (sizeChunks >> L)^2 entries, the atlas layer (-1: none);
 * - grass masks: grassMaskLayers layers of (256 / baseSpacing)^2 bytes (read as uint, 4 texels per word);
 * - the node instance buffers of TerrainPass (per frame in flight, host visible) and the staging buffers.
 * Every buffer of the set is checked against the storage-buffer limit (a larger layout is a fatal error).
 *
 * Uploads: queued on the CPU (tiles and pyramid by their shared heights, patches as copies) and recorded in queue
 * order in the transfer section, at most RenderSceneParameters::terrainUploadBytesPerFrame per frame through the
 * atlas's own UploadBatch and staging buffer per frame in flight (created once, never grows). A tile's page-table entry
 * is uploaded in the same frame as its layer; a tile counts as GPU-resident from the frame its copy is recorded, the
 * pyramid when all its rows are recorded. A patch issued while its tile's full copy is still queued lands behind it.
 *
 * Node bounds (CPU, for every CDLOD node from the leaf size up to the 4096 m roots): own = min/max of the samples the
 * node draws at its level (its tile, or the pyramid for levels >= streamedLevels; invalid without a tile), bound = own
 * united with the bounds of its children, so a parent's box contains what its resident descendants draw. Tiles,
 * removals and patches update them up to the root (patches only widen).
 *
 * Rebuild (D5): a pyramid of another space or with another layout drops every tile, mask and queued upload, hands the
 * old buffers to the deferred deletion (frames in flight still read them) and allocates the new set; a pyramid of the
 * same space and layout replaces the pyramid data in place. clear() drops the set without allocating a new one.
 */
class SAIGA_VULKAN_API TerrainAtlas
{
   public:
    // The node instances per frame (TerrainPass: then clipped)
    static constexpr int maxNodes             = 8192;
    static constexpr size_t nodeInstanceBytes = 32;
    // The staging buffer of a frame in flight: the cap plus this margin (alignment)
    static constexpr size_t stagingMargin = 256;

    struct Stats
    {
        bool hasSet   = false;
        SpaceId space = 0;
        TerrainLayout layout;
        uint64_t rebuilds = 0;
        // Per streamed level (index = level): tiles set, tiles GPU-resident
        std::array<int, 8> tilesSet{}, tilesResident{};
        int masksSet = 0, masksResident = 0;
        bool pyramidResident = false;
        // The bytes still queued, recorded in the last recorded frame
        size_t bytesQueued = 0, bytesRecorded = 0;
        uint64_t patchesApplied = 0, patchesIgnored = 0;
        // The allocated set (GPU) and RenderScene's CPU side of it
        TerrainMemory memory;
    };

    void init(VulkanBase& base, const RenderSceneParameters& params, uint64_t storageLimit);
    void destroy();
    ~TerrainAtlas() { destroy(); }

    // ---- RenderScene's terrain calls (main thread; effective in the next recorded frame)
    void setPyramid(SpaceId space, const TerrainPyramidData& pyramid);
    // New: a free layer of its level (asserted); already set: replaced in the same layer
    void setTile(const RegionId& region, const TerrainTileData& tile);
    void removeTile(const RegionId& region);
    void setGrassMask(const ChunkId& chunk, const GrassMaskData& mask);
    void removeGrassMask(const ChunkId& chunk);
    // Drops the set: nothing is drawn until the next pyramid
    void clear();
    // (x0, z0): the first sample of the sub-rectangle in the tile's (pyramid's) own sample grid, w x h samples,
    // row-major (row = z), absolute heights; the rectangle lies inside (asserted). Copied at the call. Without a layer
    // (not set, removed) or for another space than the set's: ignored and counted (Stats::patchesIgnored)
    void patchTile(const RegionId& region, int x0, int z0, int w, int h, const int32_t* heights);
    void patchPyramid(SpaceId space, int x0, int z0, int w, int h, const int32_t* heights);
    // The bytes queued and not yet recorded
    size_t uploadsPending() const { return queuedBytes; }

    // The transfer section of a frame: records the queued uploads up to the cap (the frame slot's staging buffer)
    void recordUploads(vk::CommandBuffer cmd, uint32_t frameIndex);

    // ---- the set
    bool hasSet() const { return setAllocated; }
    SpaceId space() const { return setSpace; }
    const TerrainLayout& layout() const { return setLayout; }
    std::optional<int32_t> seaLevel() const { return setSeaLevel; }
    bool pyramidResident() const { return pyramidGpu; }
    // Counts the recorded frames that changed the pyramid on the GPU (its rows or a patch; S36a2: the horizon map's
    // max-height chain and maps follow it)
    uint64_t pyramidVersion() const { return pyramidChanges; }
    // A tile of a streamed level is set (CPU) / GPU-resident (its copy recorded)
    bool tileSet(int level, int32_t tx, int32_t tz) const;
    bool tileResident(int level, int32_t tx, int32_t tz) const;
    // The mask layer of a chunk with a GPU-resident mask, else -1
    int maskLayer(int32_t cx, int32_t cz) const;
    // The page-table offset of a streamed level (entries)
    uint32_t pageOffset(int level) const { return pageOffsets[size_t(level)]; }
    // An estimate of the surface height (m) at (x, z), clamped into the space: bilinear in the finest set tile that
    // contains it (the heights as set, without later patches), else in the pyramid; none without a set. Continuous
    // within one level (TerrainPass's LOD metric: the camera height above the surface under it)
    std::optional<double> surfaceHeightEstimate(double x, double z) const;

    // ---- node bounds (height units; lo > hi: invalid)
    struct Bound
    {
        int32_t lo = 1, hi = 0;
        bool valid() const { return lo <= hi; }
    };
    int nodesPerSide(int level) const;
    Bound ownBound(int level, int32_t nx, int32_t nz) const;
    Bound bound(int level, int32_t nx, int32_t nz) const;

    // ---- buffers (TerrainPass)
    Buffer& atlasBuffer() { return atlas; }
    Buffer& pyramidBuffer() { return pyramid; }
    Buffer& pageBuffer() { return pages; }
    Buffer& maskBuffer() { return masks; }
    Buffer& nodeBuffer(uint32_t frameIndex) { return nodeBuffers[frameIndex]; }

    // ---- tests: copy a layer / the pyramid back to the host (waits for the device; after the queue is recorded)
    std::vector<int32_t> readLayerForTests(const RegionId& region);
    std::vector<int32_t> readPyramidForTests();

    // The memory of a set of this layout (the staging from the parameters' cap)
    static TerrainMemory memoryEstimate(const TerrainLayout& layout,
                                        const RenderSceneParameters& params = RenderSceneParameters());
    const Stats& stats() const { return statistics; }

   private:
    enum class Kind : uint8_t
    {
        Tile,
        Page,
        Patch,
        PyramidRows,
        PyramidPatch,
        Mask
    };
    struct Upload
    {
        Kind kind = Kind::Tile;
        RegionId region;
        int32_t cx = 0, cz = 0;  // masks
        int layer = -1;
        // Page: the entry and its value
        uint32_t pageIndex = 0;
        int32_t pageValue  = -1;
        std::shared_ptr<const std::vector<int32_t>> heights;
        std::shared_ptr<const std::vector<uint8_t>> mask;
        // Patch / PyramidPatch: the copied rows; PyramidRows: the row range
        std::vector<int32_t> patch;
        int x0 = 0, z0 = 0, w = 0, h = 0;
        int row0 = 0, rows = 0;
        // The staging footprint (16-byte aligned copies)
        size_t bytes = 0;
    };
    struct TileRecord
    {
        RegionId region;
        TerrainTileData data;
        bool used = false, gpu = false;
    };
    struct MaskRecord
    {
        int32_t cx = 0, cz = 0;
        bool used = false, gpu = false;
    };
    struct NodeBounds
    {
        int32_t ownLo = 1, ownHi = 0, lo = 1, hi = 0;
    };

    VulkanBase* base = nullptr;
    RenderSceneParameters params;
    uint64_t storageLimit = 0;
    bool initialized      = false;
    bool setAllocated     = false;
    SpaceId setSpace      = 0;
    TerrainLayout setLayout;
    std::optional<int32_t> setSeaLevel;
    bool pyramidGpu         = false;
    uint64_t pyramidChanges = 0;
    bool pageClearPending   = false;
    std::shared_ptr<const std::vector<int32_t>> pyramidHeights;
    Buffer atlas, pyramid, pages, masks;
    std::array<Buffer, 2> nodeBuffers, staging;
    UploadBatch batch;
    std::deque<Upload> queue;
    size_t queuedBytes = 0;
    std::vector<uint32_t> pageOffsets;
    // Per level: the layer (index into tiles of the level) by tile index, -1 none
    std::vector<std::vector<int32_t>> tileSlots;
    // Per level: tilesPerLevel records (layer = level * tilesPerLevel + slot)
    std::vector<std::vector<TileRecord>> tiles;
    std::map<std::pair<int32_t, int32_t>, int> maskByChunk;
    std::vector<MaskRecord> maskRecords;
    std::vector<std::vector<NodeBounds>> nodes;
    Stats statistics;

    void allocateSet(SpaceId space, const TerrainPyramidData& pyramid);
    void dropSet();
    void enqueue(Upload u);
    // Drops the queued uploads of a tile (its copy and its patches)
    void dropQueuedTile(const RegionId& region);
    void queuePyramidRows();
    size_t layerBytes() const { return size_t(setLayout.samplesPerSide()) * setLayout.samplesPerSide() * 4; }
    size_t maskLayerBytes() const { return size_t(setLayout.grassMaskSize()) * setLayout.grassMaskSize(); }
    Buffer& createSetBuffer(Buffer& b, size_t bytes, const char* what);
    TileRecord* findTile(const RegionId& region);
    // Bounds: own of the nodes of a tile (from its data, or invalid), of every pyramid level, then the bounds up.
    // widen: the union with the current own bounds (a replacement of GPU-resident data: until its copy is recorded the
    // GPU holds the old samples, and the pyramid rows of both while they are recorded)
    void tileOwnBounds(int level, int32_t tx, int32_t tz, const TerrainTileData* data, bool widen = false);
    void pyramidOwnBounds(bool widen = false);
    void widenOwn(int level, int32_t nx0, int32_t nz0, int32_t nx1, int32_t nz1, int32_t lo, int32_t hi);
    void updateBoundsUp(int level, int32_t nx0, int32_t nz0, int32_t nx1, int32_t nz1);
    NodeBounds& node(int level, int32_t nx, int32_t nz)
    {
        return nodes[size_t(level)][size_t(nz) * size_t(nodesPerSide(level)) + size_t(nx)];
    }
    std::vector<int32_t> download(Buffer& b, size_t offset, size_t bytes);
    void updateMemoryStats();
};

}  // namespace Vulkan
}  // namespace Saiga
