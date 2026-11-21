/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/Tombstones.h"

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/content/GeneratorList.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <limits>

namespace Saiga
{
namespace Engine
{
int64_t instanceIndex(const ChunkPlacement& placement, const InstanceKey& key)
{
    const auto& v = placement.instances;
    auto it       = std::lower_bound(v.begin(), v.end(), key,
                                     [](const TierOneInstance& t, const InstanceKey& k) { return t.key < k; });
    return it != v.end() && it->key == key ? int64_t(it - v.begin()) : -1;
}

const TierOneInstance* findInstance(const ChunkPlacement& placement, const InstanceKey& key)
{
    const int64_t i = instanceIndex(placement, key);
    return i < 0 ? nullptr : &placement.instances[size_t(i)];
}

int32_t tierOneSurface(const ChunkId& chunk, int spacing, int32_t x, int32_t z,
                       const std::function<int32_t(int64_t, int64_t)>& sample)
{
    // As generateChunkPlacement: the base cell of the position and its fraction (a cell is 2^fracBits units)
    const int64_t cell = int64_t(spacing) * 256;
    const int fracBits = spacing == 1 ? 8 : 9;
    const int64_t S    = 256 / spacing;
    const int64_t i = x / cell, j = z / cell;
    const int64_t fx = x - i * cell, fz = z - j * cell;
    const int64_t gi = int64_t(chunk.x) * S + i, gk = int64_t(chunk.z) * S + j;
    return terrainSurfaceUnits(sample(gi, gk), sample(gi + 1, gk), sample(gi, gk + 1), sample(gi + 1, gk + 1), fx, fz,
                               fracBits);
}

int32_t tierOneSurface(const World& world, const ChunkId& chunk, int32_t x, int32_t z)
{
    const SpaceId space = chunk.space;
    return tierOneSurface(chunk, world.spaceTemplate(space).baseSpacing, x, z,
                          [&](int64_t i, int64_t k) { return world.terrainSample(space, i, k); });
}

void reserveEditTombstones(World& world, SpaceId space, const std::vector<ChunkId>& chunks, const SampleRect& box,
                           const std::function<int32_t(int64_t, int64_t)>& postSample, EditTombstones& out)
{
    out                      = EditTombstones();
    const SpaceGeneration& g = world.generation().space(space);
    const auto& slots        = world.generators().find(g.generatorId)->slots;
    bool any                 = false;
    for (const GeneratorSlot& s : slots) any = any || (s.flags & TombstoneOnEdit);
    if (!any || !g.hasTerrain) return;
    const int spacing  = g.settings.baseSpacing;
    const int64_t cell = int64_t(spacing) * 256;
    // The changed box grown by one sample, in 1/256 m of the space
    const int64_t x0 = (box.i0 - 1) * cell, x1 = (box.i1 + 1) * cell;
    const int64_t z0 = (box.k0 - 1) * cell, z1 = (box.k1 + 1) * cell;
    const ChunkRecords& records = world.contentRecords();
    for (const ChunkId& c : chunks)
    {
        const uint16_t version                        = world.effectivePlacementVersion(c);
        const std::shared_ptr<const ChunkPlacement> p = world.generation().chunkPlacement(c, version);
        const ChunkRecord* r                          = records.find(c);
        size_t n                                      = 0;
        for (const TierOneInstance& t : p->instances)
        {
            if (!(slots[t.key.generatorSlot].flags & TombstoneOnEdit) || (r && r->tombstoned(t.key))) continue;
            const int64_t gx = int64_t(c.x) * 65536 + t.x, gz = int64_t(c.z) * 65536 + t.z;
            if (gx < x0 || gx > x1 || gz < z0 || gz > z1) continue;
            if (tierOneSurface(c, spacing, t.x, t.z, postSample) == t.y) continue;
            out.keys.push_back({c, t.key});
            n++;
        }
        if (n == 0) continue;
        out.versions.push_back({c, version});
        if (r)
            world.contentRecordsForUpdate().reserveKeys(c, n);
        else
            out.records.push_back(records.makeRecord(c, version, n));
    }
    // The send-list entries behind the edit's TerrainModification
    if (!out.keys.empty()) world.replication().reserveTombstones(space, out.keys.size(), true);
}

void commitEditTombstones(World& world, EditTombstones& reserved,
                          std::vector<std::pair<ChunkId, InstanceKey>>& tombstoned)
{
    if (reserved.keys.empty()) return;
    ChunkRecords& records = world.contentRecordsForUpdate();
    TerrainEditStore& st  = world.terrainEditsForUpdate();
    Replication& repl     = world.replication();
    for (auto& node : reserved.records) records.insertRecord(std::move(node));
    for (const auto& [c, k] : reserved.keys)
    {
        records.insert(c, k);
        st.bumpSequence(c);
        repl.appendTombstone(c, records.find(c)->placementVersion, st.sequence(c), k);
    }
    tombstoned.swap(reserved.keys);
    for (const auto& cv : reserved.versions) world.onRecordChanged(cv.first);
}

bool checkTombstone(const World& world, const ChunkId& chunk, const InstanceKey& key, std::string* error)
{
    auto fail = [&](const std::string& m)
    {
        if (error) *error = m;
        return false;
    };
    const std::string where = "space " + std::to_string(chunk.space) + " chunk (" + std::to_string(chunk.x) + ", " +
                              std::to_string(chunk.z) + ")";
    if (!world.isHost()) return fail("tombstone on a client (tombstones are host commands, R-NET-2)");
    if (chunk.space >= world.spaceCount()) return fail("tombstone: unknown space " + std::to_string(chunk.space));
    const SpaceGeneration& g = world.generation().space(chunk.space);
    if (chunk.x < 0 || chunk.z < 0 || chunk.x >= g.settings.sizeChunks || chunk.z >= g.settings.sizeChunks)
        return fail("tombstone: " + where + " is outside the space (" + std::to_string(g.settings.sizeChunks) + " x " +
                    std::to_string(g.settings.sizeChunks) + " chunks)");
    const size_t slots = world.generators().find(g.generatorId)->slots.size();
    if (key.generatorSlot >= slots)
        return fail("tombstone: generator '" + g.generatorName + "' has no slot " + std::to_string(key.generatorSlot));
    const uint32_t count = g.generator->candidatesPerChunk(key.generatorSlot);
    if (key.counter >= count)
        return fail("tombstone: counter " + std::to_string(key.counter) + " of slot " +
                    std::to_string(key.generatorSlot) + " is out of range (" + std::to_string(count) +
                    " candidates per chunk)");
    const uint16_t version                        = world.effectivePlacementVersion(chunk);
    const std::shared_ptr<const ChunkPlacement> p = world.generation().chunkPlacement(chunk, version);
    if (!findInstance(*p, key))
        return fail("tombstone: candidate " + std::to_string(key.counter) + " of slot " +
                    std::to_string(key.generatorSlot) + " in " + where + " is rejected or excluded at placement v" +
                    std::to_string(version));
    return true;
}

bool applyTombstone(World& world, const ChunkId& chunk, const InstanceKey& key)
{
    ChunkRecords& records = world.contentRecordsForUpdate();
    if (const ChunkRecord* r = records.find(chunk))
        if (r->tombstoned(key)) return false;
    // Reserve: the record (current version if new), the key capacity, S22's entry, the send-list slot
    TerrainEditStore& st = world.terrainEditsForUpdate();
    // An exhausted sequence (UINT32_MAX) cannot be bumped: refused, nothing changes (as an edit of the chunk, S22)
    if (st.sequence(chunk) == std::numeric_limits<uint32_t>::max())
    {
        SAIGA_LOG("content", "tombstone: the sequence of chunk (%d, %d) of space %u is exhausted (refused)", chunk.x,
                  chunk.z, unsigned(chunk.space));
        return false;
    }
    TerrainEditStore::EntryNode entry;
    if (!st.entry(chunk)) entry = st.makeEntry(chunk);
    Replication& repl = world.replication();
    repl.reserveTombstones(chunk.space, 1);
    records.reserve(chunk, world.effectivePlacementVersion(chunk), 1);
    // Commit
    if (!entry.empty()) st.insertEntry(std::move(entry));
    records.insert(chunk, key);
    st.bumpSequence(chunk);
    repl.appendTombstone(chunk, records.find(chunk)->placementVersion, st.sequence(chunk), key);
    world.onRecordChanged(chunk);
    return true;
}

}  // namespace Engine
}  // namespace Saiga
