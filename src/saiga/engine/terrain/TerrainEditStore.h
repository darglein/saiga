/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainEditStore (docs/engine_design.md §12.1, §12.2; S22), owned by the World (World::terrainEdits()): the terrain
// modification state of every space. The deltas (edited height minus the generated base) once per space sample in
// 16 x 16 TerrainEditBlocks keyed by sample (one copy per sample, chunk borders shared), allocated only for a non-zero
// delta that differs from the current one; the base is never stored (TerrainSystem::sample, the lazy base). Per chunk
// with state a TerrainChunkEdits entry (sequence > 0) in ChunkId order. Never evicted.
//
// Mutated only on the main thread at phase flushes, in phase-1 message handlers and between steps; the const reads are
// lock-free and safe from parallelEach bodies while nothing mutates the store (except contentHash, main thread only).
// Writers follow reserve-then-commit: reserveBlock / makeEntry allocate (before the first write), insertEntry /
// writeDelta / bumpSequence / setSequence commit without allocating.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/terrain/TerrainChunkEdits.h"
#include "saiga/engine/terrain/TerrainEditBlock.h"
#include "saiga/engine/world/SpaceTemplate.h"
#include "saiga/export.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include <unordered_map>

namespace Saiga
{
namespace Engine
{
class SAIGA_ENGINE_API TerrainEditStore
{
   public:
    using ChunkMap  = std::map<ChunkId, TerrainChunkEdits>;
    using EntryNode = ChunkMap::node_type;

    // Host edits (applyTerrainEdit)
    struct Stats
    {
        uint64_t edits = 0, noopEdits = 0, changedSamples = 0, chunksEdited = 0;
    };

    // One entry per space (SpaceId = index); the open-world spaces' sample grid (enclosed spaces hold nothing)
    // counterStart: the first value of the delta-change counter (a load's staged store continues the world's counter)
    explicit TerrainEditStore(const std::vector<SpaceTemplate>& templates, uint64_t counterStart = 0);
    TerrainEditStore(const TerrainEditStore&)            = delete;
    TerrainEditStore& operator=(const TerrainEditStore&) = delete;

    // ---- the sample grid of a space
    size_t spaceCount() const { return spaces.size(); }
    int spacing(SpaceId space) const { return spaces[space].spacing; }
    // Samples per chunk side minus one (S = 256 / spacing); a chunk has (S + 1)^2 samples
    int64_t chunkSamples(SpaceId space) const { return spaces[space].S; }
    // The last sample index of the space (sizeChunks * S)
    int64_t maxSample(SpaceId space) const { return int64_t(spaces[space].sizeChunks) * spaces[space].S; }
    int32_t sizeChunks(SpaceId space) const { return spaces[space].sizeChunks; }
    // The chunks whose closed range contains sample (i, k) (1, 2 or 4 inside the space)
    void chunksOfSample(SpaceId space, int64_t i, int64_t k, std::vector<ChunkId>& out) const;

    // ---- reads (lock-free, const)
    static uint64_t blockKey(int64_t i, int64_t k) { return uint64_t(k >> 4) << 32 | uint64_t(i >> 4); }
    // The delta of sample (i, k); 0 where no block is allocated (and outside the space)
    int32_t delta(SpaceId space, int64_t i, int64_t k) const
    {
        const Space& s = spaces[space];
        if (s.blocks.empty() || i < 0 || k < 0) return 0;
        auto it = s.blocks.find(blockKey(i, k));
        return it == s.blocks.end() ? 0 : it->second->delta[(k & 15) * 16 + (i & 15)];
    }
    // The block that holds sample (i, k), null if none is allocated
    const TerrainEditBlock* block(SpaceId space, int64_t i, int64_t k) const;
    size_t blockCount(SpaceId space) const { return spaces[space].blocks.size(); }
    // Non-zero deltas of the whole space (each sample once): the D1 early return of TerrainSystem::height
    uint64_t nonZeroSamples(SpaceId space) const { return spaces[space].nonZero; }

    // ---- chunk entries
    // Null if the chunk has no state
    const TerrainChunkEdits* entry(const ChunkId& chunk) const;
    uint32_t sequence(const ChunkId& chunk) const;
    uint64_t revision(const ChunkId& chunk) const;
    // A terrain record (E10): the chunk has non-zero deltas in its closed range
    bool hasDeltas(const ChunkId& chunk) const;
    // The chunks with sequence > 0 of a space, in ChunkId order
    const ChunkMap& chunks(SpaceId space) const { return spaces[space].chunks; }
    // Chunks with state over all spaces
    size_t chunkCount() const;
    // The dense N x N deltas of the chunk's closed range (row-major, row = z)
    void denseDelta(const ChunkId& chunk, std::vector<int32_t>& out) const;
    // The smallest and largest delta of the chunk's closed range, both including 0
    void deltaRange(const ChunkId& chunk, int32_t& lo, int32_t& hi) const;
    // FNV-1a 64 of the dense delta of a chunk with state (cached; recomputed when dirty). Main thread only: unlike the
    // other const reads it writes the cache, so it is not safe from parallelEach bodies
    uint64_t contentHash(const ChunkId& chunk) const;
    // The store's delta-change counter (monotone, never reset)
    uint64_t changeCounter() const { return counter; }

    // ---- writes (main thread)
    // Reserve: allocates the block that holds sample (i, k) if missing (zeroed: invisible until written)
    void reserveBlock(SpaceId space, int64_t i, int64_t k);
    bool blockAllocated(SpaceId space, int64_t i, int64_t k) const { return block(space, i, k) != nullptr; }
    // Reserve for a later writeDelta(space, i, k, value): allocates the block only if value is non-zero and differs
    // from the current delta (the allocation rule of every path, §12.1)
    void reserveFor(SpaceId space, int64_t i, int64_t k, int32_t value)
    {
        if (value == delta(space, i, k)) return;
        // S25: a block shared with a frozen commit is cloned before any write (also a write of 0)
        if (value != 0 || blockShared(space, i, k)) reserveBlock(space, i, k);
    }
    // S25: the block of sample (i, k) is referenced by a frozen commit
    bool blockShared(SpaceId space, int64_t i, int64_t k) const
    {
        const Space& s = spaces[space];
        auto it        = s.blocks.find(blockKey(i, k));
        return it != s.blocks.end() && it->second.use_count() > 1;
    }
    // S25: the blocks of a chunk's closed range as shared references (a commit's freeze: no copy in the step) and the
    // dense delta built from them later on any thread
    struct FrozenTerrain
    {
        int64_t i0 = 0, k0 = 0, N = 0;
        std::vector<std::pair<uint64_t, std::shared_ptr<const TerrainEditBlock>>> blocks;  // sorted by key
        void dense(std::vector<int32_t>& out) const;
    };
    void freeze(const ChunkId& chunk, FrozenTerrain& out) const;
    // Reserve: an entry node for a chunk without one (allocates; inserted later with insertEntry)
    EntryNode makeEntry(const ChunkId& chunk) const;
    // Commit: inserts a node of makeEntry (no allocation; its nonZeroSamples recomputed by a dense scan, sequence 0
    // until bumpSequence / setSequence). The chunk must have no entry
    void insertEntry(EntryNode&& node);
    // Commit: the delta of sample (i, k). A changed value updates nonZeroSamples, revision and hashDirty of every chunk
    // entry whose closed range contains the sample. A non-zero value that differs from the current one needs its
    // block (reserveBlock; asserted); writing the current value is a no-op. Returns whether the value changed
    bool writeDelta(SpaceId space, int64_t i, int64_t k, int32_t value);
    // sequence + 1; creates the entry if missing (allocates: callers that must not allocate insert a reserved entry
    // first). Host edits and S23b's tombstones; does not change revision
    void bumpSequence(const ChunkId& chunk);
    // The sequence of a client message or a load (> 0; creates the entry if missing)
    void setSequence(const ChunkId& chunk, uint32_t sequence);
    // Drops every block and entry (a load's replace, a client Resync); the counter keeps counting
    void clear();
    // Drops the entries and blocks of one space
    void clearSpace(SpaceId space);
    // A load's commit: the blocks and entries of a staged store (built for the same templates, its counter started at
    // this one's) replace this one's; no allocation. The statistics stay
    void replaceState(TerrainEditStore&& staged);

    Stats& statsForUpdate() { return statistics; }
    const Stats& stats() const { return statistics; }
    // The bytes of the blocks and entries (approximate: payload plus node overhead)
    size_t memoryBytes() const;

   private:
    struct Space
    {
        // Move only: MSVC's unordered_map move constructor is not noexcept, so std::vector would copy a copyable Space
        Space()                        = default;
        Space(const Space&)            = delete;
        Space& operator=(const Space&) = delete;
        Space(Space&&)                 = default;
        Space& operator=(Space&&)      = default;

        int spacing        = 1;
        int64_t S          = 256;
        int32_t sizeChunks = 0;
        // S25: shared (copy-on-write): a commit's freeze holds references; the reserve step clones a shared block
        std::unordered_map<uint64_t, std::shared_ptr<TerrainEditBlock>> blocks;
        ChunkMap chunks;
        uint64_t nonZero = 0;
    };
    std::vector<Space> spaces;
    uint64_t counter = 0;
    Stats statistics;
    // nonZeroSamples of an entry by a dense scan of its closed range
    uint32_t countNonZero(const ChunkId& chunk) const;
};

}  // namespace Engine
}  // namespace Saiga
