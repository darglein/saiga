/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainChunkEdits (docs/engine_design.md §12.1; S22): the modification state of one chunk in TerrainEditStore. An
// entry exists iff sequence > 0 (a chunk with state). The sequence counts the applied terrain modifications that
// changed a sample of the chunk's closed N x N range (and S23b's tombstones, through TerrainEditStore::bumpSequence);
// nonZeroSamples the non-zero deltas of the closed range (a chunk "has deltas", a terrain record, iff it is > 0);
// revision the store's delta-change counter at the last change of a delta of the closed range (local visual
// bookkeeping, F1: not saved, hashed or replicated); contentHash the FNV-1a 64 of the dense delta (lazy, hashDirty).

#pragma once

#include <cstdint>

namespace Saiga
{
namespace Engine
{
struct TerrainChunkEdits
{
    uint32_t sequence       = 0;
    uint32_t nonZeroSamples = 0;
    uint64_t revision       = 0;
    // Cached by TerrainEditStore::contentHash (a const read: mutable)
    mutable uint64_t contentHash = 0;
    mutable bool hashDirty       = true;
};

}  // namespace Engine
}  // namespace Saiga
