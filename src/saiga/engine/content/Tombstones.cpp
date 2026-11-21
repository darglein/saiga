/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/Tombstones.h"

#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/world/World.h"

#include <algorithm>

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
    TerrainEditStore::EntryNode entry;
    if (!st.entry(chunk)) entry = st.makeEntry(chunk);
    records.reserve(chunk, world.effectivePlacementVersion(chunk), 1);
    // Commit
    if (!entry.empty()) st.insertEntry(std::move(entry));
    records.insert(chunk, key);
    st.bumpSequence(chunk);
    world.onRecordChanged(chunk);
    return true;
}

}  // namespace Engine
}  // namespace Saiga
