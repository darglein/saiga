/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Tombstones (docs/engine_design.md §12.1, §12.5; S23b): the host's removal of one generated tier-1 instance, behind
// World::tombstone (the command at a phase flush, at once between steps, the world API). checkTombstone validates a
// key against the space (chunk range, slot, counter) and the chunk's placement at its effective version (generated
// synchronously through the cache: an accepted, not excluded candidate); applyTombstone writes it (record, key, S22's
// sequence, the host send-list entry, onRecordChanged). Runs serially on the main thread (R-PAR-1).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/content/ChunkContent.h"
#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/terrain/SampleRect.h"
#include "saiga/export.h"

#include <functional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

// The instance of a key in a placement (instances sorted by key; binary search); null if the key is not an accepted,
// not excluded candidate
SAIGA_ENGINE_API const TierOneInstance* findInstance(const ChunkPlacement& placement, const InstanceKey& key);
// The index of the key in the placement's instances, -1 if none
SAIGA_ENGINE_API int64_t instanceIndex(const ChunkPlacement& placement, const InstanceKey& key);

// The surface height (height units) of a tier-1 position (x, z in 1/256 m, chunk relative) with the integer surface
// rule of chunk generation over the samples of its base cell (S23b, R-CONTENT-1): sample(i, k) returns the global
// sample
SAIGA_ENGINE_API int32_t tierOneSurface(const ChunkId& chunk, int spacing, int32_t x, int32_t z,
                                        const std::function<int32_t(int64_t, int64_t)>& sample);
// The same over World::terrainSample (the current, edited surface)
SAIGA_ENGINE_API int32_t tierOneSurface(const World& world, const ChunkId& chunk, int32_t x, int32_t z);

// Tombstone-on-edit (§12.2; S23b): the keys an edit tombstones and what their commit needs, reserved in S22's reserve
// step (applyTerrainEdit step 4) and written in its commit step 5
struct EditTombstones
{
    // ChunkId then key order
    std::vector<std::pair<ChunkId, InstanceKey>> keys;
    std::vector<std::pair<ChunkId, uint16_t>> versions;  // per chunk with keys: the effective placement version
    std::vector<ChunkRecords::Map::node_type> records;   // new records
};
// For every changed chunk of a space whose generator has TombstoneOnEdit slots: the chunk's placement at its effective
// version (synchronous, cache), its non-tombstoned TombstoneOnEdit instances in the changed box grown by one sample,
// their y from the post-edit samples (postSample) compared with the generated y. Reserves the records, the key
// capacity and the send-list entries. May throw (generation): nothing is written
SAIGA_ENGINE_API void reserveEditTombstones(World& world, SpaceId space, const std::vector<ChunkId>& chunks,
                                            const SampleRect& box,
                                            const std::function<int32_t(int64_t, int64_t)>& postSample,
                                            EditTombstones& out);
// Writes them (after the edit's TerrainModification entry): records, keys, sequences, send-list entries, then
// onRecordChanged per chunk; no allocation
SAIGA_ENGINE_API void commitEditTombstones(World& world, EditTombstones& reserved,
                                           std::vector<std::pair<ChunkId, InstanceKey>>& tombstoned);

// The checks of World::tombstone without stopping: false with the error (host, space, chunk range, slot, counter, the
// accepted candidate at the effective placement version)
SAIGA_ENGINE_API bool checkTombstone(const World& world, const ChunkId& chunk, const InstanceKey& key,
                                     std::string* error);
// Applies a checked tombstone on the host; false (nothing changed: no sequence, message or notification) if the key was
// tombstoned already
SAIGA_ENGINE_API bool applyTombstone(World& world, const ChunkId& chunk, const InstanceKey& key);

}  // namespace Engine
}  // namespace Saiga
