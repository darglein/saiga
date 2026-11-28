/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The chunk sync messages (docs/engine_design.md §13.2; S26): raw engine messages with the replication epoch,
// little-endian fixed-width integers and varints (WireFormat.h).
//
//   ChunkInterest  client -> host, Control: flags (u8: bit 0 all, D19); epoch (u32); space (u16); without all: remove
//                  count (u16), add count (u16), removes: chunk x, z (i32), adds: chunk x, z (i32), ticket (u32)
//   ChunkIndex     host -> client, ChunkData: epoch (u32); space (u16); count (varint); per chunk in ChunkId order:
//                  x, z (i32)
//   ChunkSnapshot  host -> client, ChunkData: flags (u8: bit 0 zstd body, bit 1 deltas present, bit 2 record present);
//                  epoch (u32); ticket (u32); space (u16); chunk x, z (i32); sequence (u32); N (u16, 0 for a space
//                  without a sample grid); the body (zstd with bit 0): [record: placement version (u16), key count
//                  (varint), keys ascending as varint differences of slot << 24 | counter (the first absolute)]
//                  [deltas: encoded length (varint), the dense N x N deltas in TerrainDeltaCodec]
//
// Bounds: an interest of 1..4096 entries (none with all), an index of at most 262144 chunks, ascending and unique;
// snapshot keys at most kMaxChunkTombstones, strictly ascending, slot <= 255, counter < 2^24; N <= 257; the codec's
// bounds; a decompressed body of at most 8 MB; unknown flag bits and bytes left over are errors. The decoders check
// what needs no world; the handlers check the rest (ranges, N, versions, slots).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
constexpr size_t kMaxInterestEntries   = 4096;
constexpr size_t kMaxIndexChunks       = 262144;
constexpr size_t kMaxChunkSnapshotBody = size_t(8) << 20;
constexpr uint8_t kSnapshotZstd        = 1;
constexpr uint8_t kSnapshotDeltas      = 2;
constexpr uint8_t kSnapshotRecord      = 4;
constexpr uint8_t kInterestAll         = 1;

struct ChunkInterestMessage
{
    bool all       = false;
    uint32_t epoch = 0;
    SpaceId space  = 0;
    struct Add
    {
        int32_t x = 0, z = 0;
        uint32_t ticket = 0;
    };
    std::vector<std::pair<int32_t, int32_t>> removes;
    std::vector<Add> adds;
};

struct ChunkIndexMessage
{
    uint32_t epoch = 0;
    SpaceId space  = 0;
    std::vector<std::pair<int32_t, int32_t>> chunks;  // (x, z) in ChunkId order
};

struct ChunkSnapshotMessage
{
    uint32_t epoch  = 0;
    uint32_t ticket = 0;
    ChunkId chunk;
    uint32_t sequence         = 0;
    uint16_t N                = 0;
    bool hasRecord            = false;
    uint16_t placementVersion = 0;
    std::vector<InstanceKey> keys;  // ascending
    bool hasDeltas = false;
    std::vector<int32_t> deltas;  // N * N with deltas
};

SAIGA_ENGINE_API void encodeChunkInterest(const ChunkInterestMessage& m, std::vector<uint8_t>& out);
SAIGA_ENGINE_API bool decodeChunkInterest(const uint8_t* data, size_t size, ChunkInterestMessage& out,
                                          std::string& error);
SAIGA_ENGINE_API void encodeChunkIndex(const ChunkIndexMessage& m, std::vector<uint8_t>& out);
SAIGA_ENGINE_API bool decodeChunkIndex(const uint8_t* data, size_t size, ChunkIndexMessage& out, std::string& error);
// zstd for a body of at least 1 KB when it is smaller
SAIGA_ENGINE_API void encodeChunkSnapshot(const ChunkSnapshotMessage& m, std::vector<uint8_t>& out);
SAIGA_ENGINE_API bool decodeChunkSnapshot(const uint8_t* data, size_t size, ChunkSnapshotMessage& out,
                                          std::string& error);

}  // namespace Engine
}  // namespace Saiga
