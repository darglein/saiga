/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/TerrainEditStore.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/core/Hash.h"

#include <algorithm>

namespace Saiga
{
namespace Engine
{
namespace
{
// The chunk coordinates (along one axis) whose closed range [c * S, c * S + S] contains sample index v: v / S, and
// v / S - 1 on a border; clipped to [0, size)
int chunkRange(int64_t v, int64_t S, int32_t size, int32_t out[2])
{
    int n           = 0;
    const int64_t c = v / S;
    if (v % S == 0 && c - 1 >= 0 && c - 1 < size) out[n++] = int32_t(c - 1);
    if (c >= 0 && c < size) out[n++] = int32_t(c);
    return n;
}
}  // namespace

TerrainEditStore::TerrainEditStore(const std::vector<SpaceTemplate>& templates, uint64_t counterStart)
    : counter(counterStart)
{
    spaces.resize(templates.size());
    for (size_t i = 0; i < templates.size(); ++i)
    {
        // An enclosed space holds nothing; its baseSpacing is ignored (not validated, may be 0): the default grid
        const int spacing    = templates[i].kind == SpaceKind::OpenWorld ? templates[i].baseSpacing : 2;
        spaces[i].spacing    = spacing;
        spaces[i].S          = 256 / spacing;
        spaces[i].sizeChunks = templates[i].sizeChunks;
    }
}

void TerrainEditStore::chunksOfSample(SpaceId space, int64_t i, int64_t k, std::vector<ChunkId>& out) const
{
    out.clear();
    const Space& s = spaces[space];
    if (i < 0 || k < 0) return;
    int32_t xs[2], zs[2];
    const int nx = chunkRange(i, s.S, s.sizeChunks, xs), nz = chunkRange(k, s.S, s.sizeChunks, zs);
    for (int a = 0; a < nz; ++a)
        for (int b = 0; b < nx; ++b)
        {
            ChunkId c;
            c.space = space;
            c.x     = xs[b];
            c.z     = zs[a];
            out.push_back(c);
        }
    std::sort(out.begin(), out.end());
}

const TerrainEditBlock* TerrainEditStore::block(SpaceId space, int64_t i, int64_t k) const
{
    if (i < 0 || k < 0) return nullptr;
    const Space& s = spaces[space];
    auto it        = s.blocks.find(blockKey(i, k));
    return it == s.blocks.end() ? nullptr : it->second.get();
}

const TerrainChunkEdits* TerrainEditStore::entry(const ChunkId& chunk) const
{
    if (chunk.space >= spaces.size()) return nullptr;
    const ChunkMap& m = spaces[chunk.space].chunks;
    auto it           = m.find(chunk);
    return it == m.end() ? nullptr : &it->second;
}

uint32_t TerrainEditStore::sequence(const ChunkId& chunk) const
{
    const TerrainChunkEdits* e = entry(chunk);
    return e ? e->sequence : 0;
}

uint64_t TerrainEditStore::revision(const ChunkId& chunk) const
{
    const TerrainChunkEdits* e = entry(chunk);
    return e ? e->revision : 0;
}

bool TerrainEditStore::hasDeltas(const ChunkId& chunk) const
{
    const TerrainChunkEdits* e = entry(chunk);
    return e && e->nonZeroSamples > 0;
}

size_t TerrainEditStore::chunkCount() const
{
    size_t n = 0;
    for (const Space& s : spaces) n += s.chunks.size();
    return n;
}

void TerrainEditStore::denseDelta(const ChunkId& chunk, std::vector<int32_t>& out) const
{
    const Space& s  = spaces[chunk.space];
    const int64_t N = s.S + 1, i0 = int64_t(chunk.x) * s.S, k0 = int64_t(chunk.z) * s.S;
    out.assign(size_t(N * N), 0);
    if (s.blocks.empty()) return;
    // Block by block: (S / 16 + 1)^2 lookups
    for (int64_t bk = k0 >> 4; bk <= (k0 + N - 1) >> 4; ++bk)
        for (int64_t bi = i0 >> 4; bi <= (i0 + N - 1) >> 4; ++bi)
        {
            const TerrainEditBlock* b = block(chunk.space, bi * 16, bk * 16);
            if (!b) continue;
            for (int64_t k = std::max(bk * 16, k0); k < std::min(bk * 16 + 16, k0 + N); ++k)
                for (int64_t i = std::max(bi * 16, i0); i < std::min(bi * 16 + 16, i0 + N); ++i)
                    out[size_t((k - k0) * N + (i - i0))] = b->delta[(k & 15) * 16 + (i & 15)];
        }
}

void TerrainEditStore::deltaRange(const ChunkId& chunk, int32_t& lo, int32_t& hi) const
{
    lo = hi         = 0;
    const Space& s  = spaces[chunk.space];
    const int64_t N = s.S + 1, i0 = int64_t(chunk.x) * s.S, k0 = int64_t(chunk.z) * s.S;
    for (int64_t bk = k0 >> 4; bk <= (k0 + N - 1) >> 4; ++bk)
        for (int64_t bi = i0 >> 4; bi <= (i0 + N - 1) >> 4; ++bi)
        {
            const TerrainEditBlock* b = block(chunk.space, bi * 16, bk * 16);
            if (!b) continue;
            for (int64_t k = std::max(bk * 16, k0); k < std::min(bk * 16 + 16, k0 + N); ++k)
                for (int64_t i = std::max(bi * 16, i0); i < std::min(bi * 16 + 16, i0 + N); ++i)
                {
                    const int32_t v = b->delta[(k & 15) * 16 + (i & 15)];
                    lo              = std::min(lo, v);
                    hi              = std::max(hi, v);
                }
        }
}

uint32_t TerrainEditStore::countNonZero(const ChunkId& chunk) const
{
    const Space& s  = spaces[chunk.space];
    const int64_t N = s.S + 1, i0 = int64_t(chunk.x) * s.S, k0 = int64_t(chunk.z) * s.S;
    uint32_t n = 0;
    for (int64_t bk = k0 >> 4; bk <= (k0 + N - 1) >> 4; ++bk)
        for (int64_t bi = i0 >> 4; bi <= (i0 + N - 1) >> 4; ++bi)
        {
            const TerrainEditBlock* b = block(chunk.space, bi * 16, bk * 16);
            if (!b) continue;
            for (int64_t k = std::max(bk * 16, k0); k < std::min(bk * 16 + 16, k0 + N); ++k)
                for (int64_t i = std::max(bi * 16, i0); i < std::min(bi * 16 + 16, i0 + N); ++i)
                    n += b->delta[(k & 15) * 16 + (i & 15)] != 0 ? 1 : 0;
        }
    return n;
}

uint64_t TerrainEditStore::contentHash(const ChunkId& chunk) const
{
    const TerrainChunkEdits* e = entry(chunk);
    SAIGA_ASSERT(e, "TerrainEditStore::contentHash: the chunk has no state");
    if (e->hashDirty)
    {
        std::vector<int32_t> dense;
        denseDelta(chunk, dense);
        Hasher h;
        for (int32_t v : dense) h.addI32(v);
        e->contentHash = h.value();
        e->hashDirty   = false;
    }
    return e->contentHash;
}

void TerrainEditStore::reserveBlock(SpaceId space, int64_t i, int64_t k)
{
    SAIGA_ASSERT(i >= 0 && k >= 0 && i <= maxSample(space) && k <= maxSample(space),
                 "TerrainEditStore: sample outside the space");
    Space& s       = spaces[space];
    const auto key = blockKey(i, k);
    if (s.blocks.count(key)) return;
    // The block first: a failed allocation leaves no null entry behind
    auto b = std::make_unique<TerrainEditBlock>();
    s.blocks.emplace(key, std::move(b));
}

TerrainEditStore::EntryNode TerrainEditStore::makeEntry(const ChunkId& chunk) const
{
    ChunkMap tmp;
    tmp.emplace(chunk, TerrainChunkEdits());
    return tmp.extract(chunk);
}

void TerrainEditStore::insertEntry(EntryNode&& node)
{
    SAIGA_ASSERT(!node.empty(), "TerrainEditStore::insertEntry: an empty node");
    const ChunkId chunk = node.key();
    Space& s            = spaces[chunk.space];
    SAIGA_ASSERT(!s.chunks.count(chunk), "TerrainEditStore::insertEntry: the chunk has an entry");
    node.mapped()                = TerrainChunkEdits();
    node.mapped().nonZeroSamples = countNonZero(chunk);
    node.mapped().revision       = node.mapped().nonZeroSamples > 0 ? counter : 0;
    s.chunks.insert(std::move(node));
}

bool TerrainEditStore::writeDelta(SpaceId space, int64_t i, int64_t k, int32_t value)
{
    Space& s = spaces[space];
    auto it  = s.blocks.find(blockKey(i, k));
    if (it == s.blocks.end())
    {
        SAIGA_ASSERT(value == 0, "TerrainEditStore::writeDelta: the block was not reserved");
        return false;
    }
    int32_t& slot = it->second->delta[(k & 15) * 16 + (i & 15)];
    if (slot == value) return false;
    const int change = (value != 0 ? 1 : 0) - (slot != 0 ? 1 : 0);
    slot             = value;
    s.nonZero        = uint64_t(int64_t(s.nonZero) + change);
    counter++;
    if (s.chunks.empty()) return true;
    int32_t xs[2], zs[2];
    const int nx = chunkRange(i, s.S, s.sizeChunks, xs), nz = chunkRange(k, s.S, s.sizeChunks, zs);
    for (int a = 0; a < nz; ++a)
        for (int b = 0; b < nx; ++b)
        {
            ChunkId c;
            c.space = space;
            c.x     = xs[b];
            c.z     = zs[a];
            auto e  = s.chunks.find(c);
            if (e == s.chunks.end()) continue;
            e->second.nonZeroSamples = uint32_t(int64_t(e->second.nonZeroSamples) + change);
            e->second.revision       = counter;
            e->second.hashDirty      = true;
        }
    return true;
}

void TerrainEditStore::bumpSequence(const ChunkId& chunk)
{
    Space& s = spaces[chunk.space];
    auto it  = s.chunks.find(chunk);
    if (it == s.chunks.end())
    {
        insertEntry(makeEntry(chunk));
        it = s.chunks.find(chunk);
    }
    it->second.sequence++;
}

void TerrainEditStore::setSequence(const ChunkId& chunk, uint32_t sequence)
{
    SAIGA_ASSERT(sequence > 0, "TerrainEditStore::setSequence: the sequence must be > 0");
    Space& s = spaces[chunk.space];
    auto it  = s.chunks.find(chunk);
    if (it == s.chunks.end())
    {
        insertEntry(makeEntry(chunk));
        it = s.chunks.find(chunk);
    }
    it->second.sequence = sequence;
}

void TerrainEditStore::clear()
{
    for (SpaceId i = 0; i < spaces.size(); ++i) clearSpace(i);
}

void TerrainEditStore::clearSpace(SpaceId space)
{
    Space& s = spaces[space];
    if (s.nonZero > 0) counter++;
    s.blocks.clear();
    s.chunks.clear();
    s.nonZero = 0;
}

void TerrainEditStore::replaceState(TerrainEditStore&& staged)
{
    SAIGA_ASSERT(staged.spaces.size() == spaces.size() && staged.counter >= counter,
                 "TerrainEditStore::replaceState: another layout or an older counter");
    spaces.swap(staged.spaces);
    // Monotone: a change of the state is a change for every reader of revisions
    counter = staged.counter + 1;
}

size_t TerrainEditStore::memoryBytes() const
{
    size_t bytes = 0;
    for (const Space& s : spaces)
    {
        bytes += s.blocks.size() * (sizeof(TerrainEditBlock) + 4 * sizeof(void*));
        bytes += s.blocks.bucket_count() * sizeof(void*);
        bytes += s.chunks.size() * (sizeof(ChunkMap::value_type) + 4 * sizeof(void*));
    }
    return bytes;
}

}  // namespace Engine
}  // namespace Saiga
