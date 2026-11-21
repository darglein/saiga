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
#include "saiga/export.h"

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

// The checks of World::tombstone without stopping: false with the error (host, space, chunk range, slot, counter, the
// accepted candidate at the effective placement version)
SAIGA_ENGINE_API bool checkTombstone(const World& world, const ChunkId& chunk, const InstanceKey& key,
                                     std::string* error);
// Applies a checked tombstone on the host; false (nothing changed: no sequence, message or notification) if the key was
// tombstoned already
SAIGA_ENGINE_API bool applyTombstone(World& world, const ChunkId& chunk, const InstanceKey& key);

}  // namespace Engine
}  // namespace Saiga
