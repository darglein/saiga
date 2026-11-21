/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The plan hashes (docs/engine_design.md §12.1; S23a, Scope 6). FNV-1a 64 (Hasher, little-endian):
//
//   plan (saiga.content.plan.v1): u64 seed, u32 pinned version, the template view (string name, u8 kind, u32 template
//     index, i32 base spacing, i32 size in chunks, u8 has sea level, i32 sea level or 0); u32 site count, per site in
//     plan order: u32 key, string prefab, u32 prefab version, u64 prefab content hash, i64 x, i64 z, u8 quarter turns,
//     i32 base height, four i32 footprint values; u32 listed-instance count, per instance in list order: u32 template
//     index, u64 seed, u32 parent, u32 site key, u8 link, u8 origin; then biomes and rivers, each i32 cell size, i32
//     originX, originZ, width, height, u32 value count, every value i32. cloudCoverage is not hashed.
//   combined (saiga.content.plans.v1): u32 space count N, per SpaceId 0..N-1: u32 template index, string generator
//     name, u64 that space's plan hash.

#pragma once

#include "saiga/engine/content/WorldPlan.h"
#include "saiga/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// prefabHashes: the content hash of each site's prefab (plan.sites order)
SAIGA_ENGINE_API uint64_t planHash(const WorldPlan& plan, uint16_t pinnedVersion,
                                   const std::vector<uint64_t>& prefabHashes);
// The hash of the template view alone (the plan cache's key)
SAIGA_ENGINE_API uint64_t templateViewHash(const GenerationTemplate& view);

struct CombinedPlanEntry
{
    uint16_t templateIndex = 0;
    std::string generator;
    uint64_t planHash = 0;
};
SAIGA_ENGINE_API uint64_t combinedPlanHash(const std::vector<CombinedPlanEntry>& spaces);

}  // namespace Engine
}  // namespace Saiga
