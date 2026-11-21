/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ChunkRecords (docs/engine_design.md §12.1, §15.3; S23a, S23b): the World's chunk records (state, saved in CREC): per
// chunk the placement version its tier-1 content is fixed at and its tombstones (sorted, unique), and the host-only set
// of the chunks whose prefab entities are stamped. A chunk has a record when the host stamped it (host only), when it
// has a tombstone (every peer), or when a record of another placement version was received or loaded; every other chunk
// has the effective record {current placement version, no tombstones}. The modification sequence of a chunk stays in
// S22's TerrainEditStore (E10). Iterated in ChunkId order. Every writer calls World::onRecordChanged after a change.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/export.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace Saiga
{
namespace Engine
{
struct ChunkRecord
{
    uint16_t placementVersion = 0;
    // Sorted by (slot, counter), unique
    std::vector<InstanceKey> tombstones;

    bool tombstoned(const InstanceKey& key) const
    {
        return std::binary_search(tombstones.begin(), tombstones.end(), key);
    }
    bool operator==(const ChunkRecord& o) const
    {
        return placementVersion == o.placementVersion && tombstones == o.tombstones;
    }
    bool operator!=(const ChunkRecord& o) const { return !(*this == o); }
};

class SAIGA_ENGINE_API ChunkRecords
{
   public:
    using Map = std::map<ChunkId, ChunkRecord>;

    const Map& records() const { return byChunk; }
    const ChunkRecord* find(const ChunkId& chunk) const;
    // Host: the chunks whose prefab entities are stamped
    const std::set<ChunkId>& stamped() const { return stampedChunks; }
    bool isStamped(const ChunkId& chunk) const { return stampedChunks.count(chunk) != 0; }
    bool empty() const { return byChunk.empty() && stampedChunks.empty(); }
    // The chunks with tombstones
    size_t tombstonedChunks() const;
    // A stamped chunk: in the stamped set, with a record at the current version if it has none
    void markStamped(const ChunkId& chunk, uint16_t currentVersion);

    // ---- reserve-then-commit writers (S23b): reserve allocates (a record node, key capacity), commit does not
    // Reserve: the record of the chunk (created with the version if missing) with capacity for `more` keys
    ChunkRecord& reserve(const ChunkId& chunk, uint16_t version, size_t more = 1);
    // Commit: inserts the key in order (after reserve: no allocation); false if it was tombstoned already
    bool insert(const ChunkId& chunk, const InstanceKey& key);
    // A load (the staged state, committed with the world)
    void replace(Map records, std::set<ChunkId> stamped);
    // A client Resync: every record goes (the stamped set is host-only and empty on a client)
    void clear();
    // Test-only (S23b): a record with the version and the keys (sorted, unique; merged into an existing record of the
    // same version); the caller calls World::onRecordChanged
    void injectForTests(const ChunkId& chunk, uint16_t placementVersion, const std::vector<InstanceKey>& keys);

    struct Stats
    {
        // Stamped chunks and entities; prefab components that could not be applied (logged)
        uint64_t chunks = 0, entities = 0, failedComponents = 0;
        // S23b: applied tombstones (host calls, tombstone-on-edit, client messages)
        uint64_t tombstones = 0;
    };
    Stats& statsForUpdate() { return statistics; }
    const Stats& stats() const { return statistics; }

   private:
    Map byChunk;
    std::set<ChunkId> stampedChunks;
    Stats statistics;
};

// Keys are sorted and unique
SAIGA_ENGINE_API bool sortedUniqueKeys(const std::vector<InstanceKey>& keys);

}  // namespace Engine
}  // namespace Saiga
