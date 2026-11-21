/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The section hashes of a chunk's content (docs/engine_design.md §12.1; S23a, Scope 6): FNV-1a 64 through Hasher, each
// section its own Hasher seeded with fnv1a64("saiga.content.<section>.v1"); u16 values hashed as u32, u8 as u8,
// strings with addString, signed values as i32 / i64. The byte layouts are fixed (a change is a new .v2 seed name and
// a new entry of the committed hash file):
//
//   heights  u32 samples per side, u32 base spacing, every sample i32 (row-major z then x); enclosed: u32 0, u32 0
//   tier1    u32 count; per instance in (slot, counter) order: u32 slot, u32 counter, u32 meshSlot, i32 x, i32 z,
//            i32 y, u32 yaw, u32 scale, u32 flags
//   grass    u32 size, every density byte (u8), row-major z then x
//   stamps   u32 count; per record in (site index, entity index) order: u32 site key, u32 entity index, string prefab,
//            u32 prefab version, string class, i64 x, i64 z (1/256 m, space coordinates), i32 y, u32 yaw
//   statics  u32 count; per part in (site index, part index) order: u32 site key, u32 part index, string mesh, i64 x,
//            i64 z, i32 y, u32 yaw
//   total    u32 space, i32 chunk x, i32 chunk z, u32 pinned version, u32 placement version, the five section hashes

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/export.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Saiga
{
namespace Engine
{
struct TierOneInstance;
struct StampRecord;
struct StaticPart;
struct GrassMask;

struct ContentHashes
{
    uint64_t heights = 0, tier1 = 0, grass = 0, stamps = 0, statics = 0;
    uint64_t total = 0;
};

// samples: null (enclosed: no height map) or the level-0 samples (samplesPerSide^2)
SAIGA_ENGINE_API uint64_t hashHeights(const std::vector<int32_t>* samples, int32_t samplesPerSide, int32_t baseSpacing);
SAIGA_ENGINE_API uint64_t hashTierOne(const std::vector<TierOneInstance>& instances);
SAIGA_ENGINE_API uint64_t hashGrass(const GrassMask& mask);
SAIGA_ENGINE_API uint64_t hashStamps(const std::vector<StampRecord>& records);
SAIGA_ENGINE_API uint64_t hashStatics(const std::vector<StaticPart>& parts);
SAIGA_ENGINE_API uint64_t hashTotal(const ChunkId& chunk, uint16_t pinnedVersion, uint16_t placementVersion,
                                    const ContentHashes& sections);

}  // namespace Engine
}  // namespace Saiga
