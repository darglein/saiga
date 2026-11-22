/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ChunkRecords (docs/engine_design.md §12.1, §15.3; S23a): the World's chunk records (state, saved in CREC): per chunk
// the placement version it keeps, and the host-only set of the chunks whose prefab entities are stamped. A chunk gets a
// record when the host stamps it (S23b adds the chunks with tombstones; a load keeps the saved records); every other
// chunk is generated with the current placement version. Iterated in ChunkId order.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/export.h"

#include <cstdint>
#include <map>
#include <set>

namespace Saiga
{
namespace Engine
{
struct ChunkRecord
{
    uint16_t placementVersion = 0;
};

class SAIGA_ENGINE_API ChunkRecords
{
   public:
    const std::map<ChunkId, ChunkRecord>& records() const { return byChunk; }
    const ChunkRecord* find(const ChunkId& chunk) const;
    // Host: the chunks whose prefab entities are stamped
    const std::set<ChunkId>& stamped() const { return stampedChunks; }
    bool isStamped(const ChunkId& chunk) const { return stampedChunks.count(chunk) != 0; }
    bool empty() const { return byChunk.empty() && stampedChunks.empty(); }
    // A stamped chunk: in the stamped set, with a record at the current version if it has none
    void markStamped(const ChunkId& chunk, uint16_t currentVersion);
    // A load (the staged state, committed with the world)
    void replace(std::map<ChunkId, ChunkRecord> records, std::set<ChunkId> stamped);
    void clear();

    struct Stats
    {
        // Stamped chunks and entities; prefab components that could not be applied (logged)
        uint64_t chunks = 0, entities = 0, failedComponents = 0;
    };
    Stats& statsForUpdate() { return statistics; }
    const Stats& stats() const { return statistics; }

   private:
    std::map<ChunkId, ChunkRecord> byChunk;
    std::set<ChunkId> stampedChunks;
    Stats statistics;
};

}  // namespace Engine
}  // namespace Saiga
