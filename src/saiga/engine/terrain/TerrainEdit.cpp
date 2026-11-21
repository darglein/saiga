/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/TerrainEdit.h"

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainStore.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace Saiga
{
namespace Engine
{
void gatherSamples(const World& world, SpaceId space, const SampleRect& rect, std::vector<int32_t>& base,
                   std::vector<int32_t>& height)
{
    const TerrainSystem& terrain = world.terrain();
    const TerrainEditStore& st   = world.terrainEdits();
    const int64_t w = rect.width(), h = rect.height();
    base.resize(size_t(w * h));
    height.resize(size_t(w * h));
    if (rect.empty()) return;
    const SpaceTemplate& t = terrain.spaceTemplate(space);
    // The store's grid (PR review: the baseSpacing of an enclosed space is ignored and may be 0)
    const int64_t S        = st.chunkSamples(space);
    const int64_t N        = S + 1;
    const TerrainStore& ts = terrain.store(space);
    for (int64_t k = rect.k0; k <= rect.k1; ++k)
    {
        int32_t* row = base.data() + size_t((k - rect.k0) * w);
        // Per level-0 tile column: a row copy of a resident tile (the far edge sample from the last tile), else the
        // base point query
        int64_t i = rect.i0;
        while (i <= rect.i1)
        {
            const int64_t tx               = std::min<int64_t>(i / S, t.sizeChunks - 1);
            const int64_t tz               = std::min<int64_t>(k / S, t.sizeChunks - 1);
            const int64_t end              = std::min(rect.i1, tx * S + S);
            const TerrainStore::Tile* tile = nullptr;
            if (terrain.hasTerrain(space))
            {
                RegionId r;
                r.space = space;
                r.level = 0;
                r.x     = int32_t(tx);
                r.z     = int32_t(tz);
                tile    = ts.tile(r);
            }
            if (tile)
            {
                const int32_t* src = tile->data.heights->data() + size_t((k - tz * S) * N + (i - tx * S));
                std::memcpy(row + (i - rect.i0), src, size_t(end - i + 1) * sizeof(int32_t));
            }
            else
                for (int64_t j = i; j <= end; ++j) row[j - rect.i0] = terrain.sample(space, j, k);
            i = end + 1;
        }
        for (int64_t j = rect.i0; j <= rect.i1; ++j)
        {
            const size_t idx = size_t((k - rect.k0) * w + (j - rect.i0));
            SAIGA_ASSERT(std::abs(base[idx]) <= kTerrainHeightLimit,
                         "gatherSamples: a base beyond kTerrainHeightLimit");
            const int64_t v = int64_t(base[idx]) + st.delta(space, j, k);
            height[idx] = int32_t(std::min<int64_t>(std::max<int64_t>(v, -kTerrainHeightLimit), kTerrainHeightLimit));
        }
    }
}

TerrainEditResult applyTerrainEdit(World& world, SpaceId space, const TerrainBrush& brush)
{
    TerrainEditResult result;
    result.space                = space;
    TerrainEditStore& st        = world.terrainEditsForUpdate();
    TerrainEditStore::Stats& ss = st.statsForUpdate();
    const int spacing           = st.spacing(space);
    const int64_t maxSample     = st.maxSample(space);
    const int64_t S             = st.chunkSamples(space);

    // 1. quantize, the affected and the read rectangle (clipped)
    const QuantizedBrush q    = BrushKernel::quantize(brush);
    const SampleRect affected = BrushKernel::affectedRect(q, spacing, maxSample);
    const SampleRect read     = BrushKernel::readRect(q, affected, maxSample);
    ss.edits++;
    if (affected.empty())
    {
        ss.noopEdits++;
        return result;
    }
    // 2. the base and the pre-edit heights (no allocation in the store)
    std::vector<int32_t> base, pre, out(size_t(affected.width() * affected.height()));
    gatherSamples(world, space, read, base, pre);
    // 3. the kernel; the changed samples and their tight box
    const uint64_t changed = BrushKernel::evaluate(q, spacing, affected, read, pre.data(), out.data());
    if (changed == 0)
    {
        ss.noopEdits++;
        return result;
    }
    const int64_t rw = read.width(), aw = affected.width();
    auto preAt  = [&](int64_t i, int64_t k) { return pre[size_t((k - read.k0) * rw + (i - read.i0))]; };
    auto baseAt = [&](int64_t i, int64_t k) { return base[size_t((k - read.k0) * rw + (i - read.i0))]; };
    auto outAt  = [&](int64_t i, int64_t k) { return out[size_t((k - affected.k0) * aw + (i - affected.i0))]; };
    SampleRect box;
    box.i0 = affected.i1 + 1;
    box.k0 = affected.k1 + 1;
    box.i1 = affected.i0 - 1;
    box.k1 = affected.k0 - 1;
    // The chunks whose closed range contains a changed sample (at most 2 x 2: the brush is strictly inside 2 * 128 m)
    std::vector<ChunkId> chunks;
    for (int64_t k = affected.k0; k <= affected.k1; ++k)
        for (int64_t i = affected.i0; i <= affected.i1; ++i)
        {
            if (outAt(i, k) == preAt(i, k)) continue;
            box.i0 = std::min(box.i0, i);
            box.i1 = std::max(box.i1, i);
            box.k0 = std::min(box.k0, k);
            box.k1 = std::max(box.k1, k);
            // The chunks of this sample: i / S, and i / S - 1 on a border (clipped to the space)
            const int64_t cx = i / S, cz = k / S;
            for (int64_t z = (k % S == 0 ? cz - 1 : cz); z <= cz; ++z)
                for (int64_t x = (i % S == 0 ? cx - 1 : cx); x <= cx; ++x)
                {
                    if (x < 0 || z < 0 || x >= st.sizeChunks(space) || z >= st.sizeChunks(space)) continue;
                    ChunkId c;
                    c.space = space;
                    c.x     = int32_t(x);
                    c.z     = int32_t(z);
                    if (std::find(chunks.begin(), chunks.end(), c) == chunks.end()) chunks.push_back(c);
                }
        }
    std::sort(chunks.begin(), chunks.end());
    // A chunk whose sequence is exhausted (UINT32_MAX, e.g. from a save) cannot be bumped (0 is not a valid sequence
    // of saves and messages): the edit is refused and changes nothing (checked before any block is reserved)
    for (const ChunkId& c : chunks)
        if (st.sequence(c) == std::numeric_limits<uint32_t>::max())
        {
            SAIGA_LOG("terrain", "editTerrain: the sequence of chunk (%d, %d) of space %u is exhausted (refused)", c.x,
                      c.z, unsigned(space));
            ss.noopEdits++;
            return result;
        }
    // 4. reserve the blocks of the new deltas (only a non-zero delta that differs from the current one)
    for (int64_t k = box.k0; k <= box.k1; ++k)
        for (int64_t i = box.i0; i <= box.i1; ++i)
            if (outAt(i, k) != preAt(i, k)) st.reserveFor(space, i, k, outAt(i, k) - baseAt(i, k));
    // 4. reserve the entries of new chunks, the parts of the send-list entry, the result and the change-list slot
    std::vector<TerrainEditStore::EntryNode> entries;
    entries.reserve(chunks.size());
    for (const ChunkId& c : chunks)
        if (!st.entry(c)) entries.push_back(st.makeEntry(c));
    std::vector<TerrainPart> parts(chunks.size());
    for (size_t n = 0; n < chunks.size(); ++n)
    {
        TerrainPart& p    = parts[n];
        const ChunkId& c  = chunks[n];
        const int64_t ci0 = int64_t(c.x) * S, ck0 = int64_t(c.z) * S;
        const int64_t x0 = std::max(box.i0, ci0), x1 = std::min(box.i1, ci0 + S);
        const int64_t z0 = std::max(box.k0, ck0), z1 = std::min(box.k1, ck0 + S);
        p.chunk = c;
        p.x0    = int(x0 - ci0);
        p.z0    = int(z0 - ck0);
        p.w     = int(x1 - x0 + 1);
        p.h     = int(z1 - z0 + 1);
        p.deltas.resize(size_t(p.w) * size_t(p.h));
        for (int64_t k = z0; k <= z1; ++k)
            for (int64_t i = x0; i <= x1; ++i)
            {
                // The new delta of every sample of the box (unchanged samples keep theirs)
                const int32_t v = outAt(i, k) != preAt(i, k) ? outAt(i, k) - baseAt(i, k) : st.delta(space, i, k);
                p.deltas[size_t((k - z0) * p.w + (i - x0))] = v;
            }
    }
    result.chunks.resize(chunks.size());
    Replication& repl = world.replication();
    repl.reserveHostState();
    world.reserveTerrainChange();
    // S23b's extension point (F2): its tombstone-on-edit is computed here from the kernel output (the post-edit
    // samples) and its chunk records, key capacity and send-list entries are reserved; nothing is written yet

    // 5. commit (writes only, no allocation): entries, deltas, sequences, parts
    for (TerrainEditStore::EntryNode& e : entries) st.insertEntry(std::move(e));
    for (int64_t k = box.k0; k <= box.k1; ++k)
        for (int64_t i = box.i0; i <= box.i1; ++i)
            if (outAt(i, k) != preAt(i, k)) st.writeDelta(space, i, k, outAt(i, k) - baseAt(i, k));
    for (size_t n = 0; n < chunks.size(); ++n)
    {
        st.bumpSequence(chunks[n]);
        parts[n].sequence = st.sequence(chunks[n]);
        result.chunks[n]  = {chunks[n], parts[n].sequence, parts[n].x0, parts[n].z0, parts[n].w, parts[n].h};
    }
    repl.appendTerrainModification(space, std::move(parts));
    // S23b writes the tombstones it reserved here (keys, bumpSequence, its send-list entries after this one)

    result.changedSamples = changed;
    result.i0             = box.i0;
    result.k0             = box.k0;
    result.i1             = box.i1;
    result.k1             = box.k1;
    // 6. the single path of every applied change
    world.onTerrainChanged(space, box);
    // 7. statistics
    ss.changedSamples += changed;
    ss.chunksEdited += chunks.size();
    return result;
}

}  // namespace Engine
}  // namespace Saiga
