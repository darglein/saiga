/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The generated content of a chunk (docs/engine_design.md §12.1; S23a), split by the version that keys it: ChunkBase
// (the pinned plan / terrain version: the level-0 heights), ChunkPlacement (the chunk's placement version: tier-1
// instances and the grass mask), and the plan-only stamp and static-part lists. Integer only (content code: the
// ContentLint test scans this header).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/content/Generator.h"
#include "saiga/engine/content/gen/SectionHash.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// Plan-version content: the level-0 heights at the template's base spacing (S20's layout: (256 / spacing + 1)^2
// samples, row-major z then x, sample (i, j) at the chunk origin + (i, j) * spacing m; open world only, null in an
// enclosed space)
struct ChunkBase
{
    ChunkId id;
    std::shared_ptr<const std::vector<int32_t>> heights;
    int32_t samplesPerSide = 0, baseSpacing = 0;
    uint64_t heightsHash = 0;
};

struct TierOneInstance
{
    InstanceKey key;  // {slot, counter}
    uint16_t meshSlot = 0;
    int32_t x = 0, z = 0;  // 1/256 m chunk relative
    int32_t y    = 0;      // height units, absolute
    uint16_t yaw = 0, scale = 1024;
    uint32_t flags = 0;
};

// Placement-version content: the tier-1 instances sorted by (slot, counter), the grass mask (cells inside an exclusion
// are 0) and every section hash with the total
struct ChunkPlacement
{
    ChunkId id;
    uint16_t pinnedVersion = 0, placementVersion = 0;
    std::vector<TierOneInstance> instances;
    GrassMask grass;
    ContentHashes hashes;
};

// A prefab entity to stamp (plan only): x, z in 1/256 m and y in height units (site.baseHeight + prefab y) in space
// coordinates; yaw = prefab yaw + quarterTurns * 16384
struct StampRecord
{
    uint32_t siteIndex = 0, siteKey = 0;
    uint32_t entity = 0;  // the prefab's entity index
    std::string prefab;
    uint16_t prefabVersion = 1;
    std::string className;
    int64_t x = 0, z = 0;
    int32_t y    = 0;
    uint16_t yaw = 0;
};

// A static part of a prefab (plan only), as StampRecord
struct StaticPart
{
    uint32_t siteIndex = 0, siteKey = 0;
    uint32_t part = 0;
    std::string mesh;
    int64_t x = 0, z = 0;
    int32_t y    = 0;
    uint16_t yaw = 0;
};

}  // namespace Engine
}  // namespace Saiga
