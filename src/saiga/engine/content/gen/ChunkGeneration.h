/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Chunk generation (docs/engine_design.md §12.1; S23a): pure and thread safe. chunkBase: the level-0 heights from the
// composed height at every sample incl. the shared border. chunkPlacement: per slot and counter the candidate,
// rejected outside the chunk (asserted) or inside an exclusion, y from the terrain surface rule of its base cell; the
// grass mask with the excluded cells cleared; the section hashes.

#pragma once

#include "saiga/engine/content/ChunkContent.h"
#include "saiga/engine/content/gen/SitePatch.h"
#include "saiga/export.h"

#include <memory>
#include <vector>

namespace Saiga
{
namespace Engine
{
// What chunk generation reads of a space (all immutable)
struct ChunkInputs
{
    const Generator* generator                                   = nullptr;
    uint16_t pinnedVersion                                       = 1;
    const WorldPlan* plan                                        = nullptr;
    SpaceId space                                                = 0;
    const std::vector<std::shared_ptr<const SiteContent>>* sites = nullptr;
    // An open-world space has a height map; an enclosed space has none (heights null, tier-1 y = 0)
    bool openWorld = true;
};

SAIGA_ENGINE_API ChunkBase generateChunkBase(const ChunkInputs& in, const ChunkId& chunk);
// stamps / statics: the hashes of the chunk's plan lists (they enter the total)
SAIGA_ENGINE_API ChunkPlacement generateChunkPlacement(const ChunkInputs& in, const ChunkBase& base,
                                                       uint16_t placementVersion, uint64_t stampsHash,
                                                       uint64_t staticsHash);

}  // namespace Engine
}  // namespace Saiga
