/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The committed content hashes (docs/engine_design.md §12.1; S23a, E12): checkContentHashes runs phase A of the
// generation pipeline (no World: no class, component or mesh checks) for every setup entry of a hash file with the
// entry's seed and versions and compares the plan hashes, the combined plan hash, the listed chunks' sections and
// totals, the digest and the toolkit hashes. The engine's file is data/tests/engine/content_hashes.json
// (ContentHash.CommittedHashesReproduce); games keep their own and call this from a scenario (S23c).
//
// File format 1:
//   {"format": 1,
//    "toolkit": {"trig_table": "0x..", "noise_grid": "0x..", "smooth_q16": "0x.."},          optional
//    "setups": [{"name": "test", "seed": "12345",                                              the seed as a string
//                "generators": {"<name>": {"plan_version": 1, "placement_version": 1, "slots": ["a", "b"]}},
//                "plans": {"<SpaceId>": {"generator": "<name>", "hash": "0x.."}},
//                "combined_plan_hash": "0x..",
//                "digest_block": [x0, z0, x1, z1],
//                "chunks": {"<space>:<x>:<z>": {"total": "0x..", "heights": .., "tier1": .., "grass": .., "stamps": ..,
//                                               "statics": ..}},
//                "digest": "0x.."}]}
// The digest: FNV-1a 64 seeded with fnv1a64("saiga.content.digest.v1") over u32 count, then per chunk (i32 x, i32 z,
// u64 total) for the chunks [x0, x1) x [z0, z1) of space 0 and then every site chunk of space 0 outside the block
// (the keys of its sitesByChunk), each once, in ChunkId order. Chunks are generated on a JobPool(3).
//
// update: the file is rewritten in place (the entries, chunk IDs and digest blocks are kept, every value recomputed)
// and the report fails (an update run never passes). A missing file is an error naming the path, also with update:
// a new file is committed first as a skeleton whose values are "0x0".

#pragma once

#include "saiga/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
struct WorldSetup;

struct SAIGA_ENGINE_API ContentHashReport
{
    bool ok = false;
    // Every mismatch: entry, chunk or plan, section, expected, actual; or an error (a missing file, a parse error)
    std::vector<std::string> problems;
    int entries = 0, chunks = 0;
    double seconds = 0;
    bool updated   = false;
    std::string summary() const;
};

SAIGA_ENGINE_API ContentHashReport checkContentHashes(const WorldSetup& setup, const std::string& file, bool update);

// A u64 from a string with std::from_chars: decimal, or hexadecimal after a "0x" prefix (hex); the whole string must
// be consumed
SAIGA_ENGINE_API bool parseU64(const std::string& text, bool hex, uint64_t& out);
SAIGA_ENGINE_API std::string hexU64(uint64_t v);

// The toolkit hashes of the file's "toolkit" entry: the trig table, a 256 x 256 noise grid at negative and positive
// coordinates (fbm, ridged, gradientNoise, valueNoise) and all 65537 smoothQ16 values
SAIGA_ENGINE_API uint64_t toolkitTrigHash();
SAIGA_ENGINE_API uint64_t toolkitNoiseHash();
SAIGA_ENGINE_API uint64_t toolkitSmoothHash();

}  // namespace Engine
}  // namespace Saiga
