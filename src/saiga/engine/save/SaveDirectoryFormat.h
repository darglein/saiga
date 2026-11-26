/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The on-disk layout of a save directory (docs/engine_design.md §15.1; S25). All integers little-endian; every file
// ends with an FNV-1a 64 of all its preceding bytes, except region files, whose records carry their own checksum.
//
//   <dir>/lock                                the lock file (SaveFileSystem::lock; ignored by the open's cleanup)
//   <dir>/header.sgh                          the header: magic SGWH, format (u32), the engine version that created
//                                             the directory (string), the schema tables (count u32, each
//                                             SchemaTable::write; append-only, numbered from 0), trailer
//   <dir>/world.<c>.sgw                       the world-state file of commit c (decimal, no padding): magic SGWF,
//                                             format (u32), commit (u64), the header tables the commit needs (u32),
//                                             two framed parts (uncompressed length u64, compressed length u64, a zstd
//                                             frame): the world-state blob (World::save without TERR, CREC, EGRP) and
//                                             the chunk index (CIDX), trailer
//   <dir>/regions/<space>/<rx>.<rz>.<g>.sgr   region file generation g of region (rx, rz) of a space: a 24-byte
//                                             header (magic SGRF, format u32, space u16, padding u16, rx, rz i32,
//                                             generation u32), then records, appended, never rewritten in place
//   <dir>/*.tmp, regions/**/*.tmp             temporary files (removed at open)
//   <dir>/world.<c>.sgw.corrupt               a world-state file that failed its checksum (kept for diagnosis)
//
// A record (ChunkRecordCodec): a 32-byte header (magic SGCR u32, record format u16, kind u8 (0 chunk, 1 space), flags
// u8, space u16, reserved u16, x, z i32, commit u64, payload length u32), the payload (one zstd frame) and an FNV-1a 64
// over header and payload. The uncompressed payload is sections in a fixed order (tag u32, format version u32, length
// u64, bytes): RTER, RREC, RGRP (chunk records), RSPC (space records).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/save/SaveFormat.h"

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

namespace Saiga
{
namespace Engine
{
constexpr uint32_t kTagDirHeader     = saveTag("SGWH");
constexpr uint32_t kTagWorldFile     = saveTag("SGWF");
constexpr uint32_t kTagRegionFile    = saveTag("SGRF");
constexpr uint32_t kTagRecord        = saveTag("SGCR");
constexpr uint32_t kTagChunkIndex    = saveTag("CIDX");
constexpr uint32_t kTagRecTerrain    = saveTag("RTER");
constexpr uint32_t kTagRecRecord     = saveTag("RREC");
constexpr uint32_t kTagRecGroups     = saveTag("RGRP");
constexpr uint32_t kTagRecSpace      = saveTag("RSPC");
constexpr uint32_t kDirHeaderFormat  = 1;
constexpr uint32_t kWorldFileFormat  = 1;
constexpr uint32_t kRegionFormat     = 1;
constexpr uint16_t kRecordFormat     = 1;
constexpr uint32_t kChunkIndexFormat = 1;
// The format version of each record section (an unknown tag or a higher version is an error naming the chunk)
constexpr uint32_t kRecTerrainVersion = 1;
constexpr uint32_t kRecRecordVersion  = 1;
constexpr uint32_t kRecGroupsVersion  = 1;
constexpr uint32_t kRecSpaceVersion   = 1;
// The header sizes
constexpr size_t kRegionHeaderSize = 24;
constexpr size_t kRecordHeaderSize = 32;
// A region is 16 x 16 chunks (4096 m, the CDLOD root node)
constexpr int32_t kStoreRegionChunks = 16;
// The largest uncompressed payload or framed part (untrusted input: checked from the zstd frame header)
constexpr uint64_t kMaxUncompressedPart = uint64_t(256) << 20;
// The largest valid world-state file: two parts (zstd's worst case stays below 1 % over the raw size) plus framing
constexpr uint64_t kMaxWorldFileBytes = 2 * kMaxUncompressedPart + (uint64_t(8) << 20);
// Two kept commits (the newest and its backup); compaction of a region file when its live bytes fall below half of
// its size and it is at least 1 MiB
constexpr uint64_t kKeptCommits     = 2;
constexpr uint64_t kCompactMinBytes = uint64_t(1) << 20;
// Record kinds
constexpr uint8_t kRecordKindChunk = 0;
constexpr uint8_t kRecordKindSpace = 1;

// The key of a space record in the chunk store (outside every chunk range: sizes are bounded to 512 chunks)
inline ChunkId spaceRecordKey(SpaceId space)
{
    ChunkId c;
    c.space = space;
    c.x     = std::numeric_limits<int32_t>::min();
    c.z     = std::numeric_limits<int32_t>::min();
    return c;
}
inline bool isSpaceRecordKey(const ChunkId& c)
{
    return c.x == std::numeric_limits<int32_t>::min() && c.z == std::numeric_limits<int32_t>::min();
}

// The region of a chunk (space records: region (0, 0))
inline int32_t regionOf(int32_t chunkCoordinate)
{
    // In 64 bits: a crafted index may carry INT32_MIN, whose negation overflows in 32 bits (the range check follows)
    const int64_t c = chunkCoordinate;
    return int32_t(c >= 0 ? c / kStoreRegionChunks : -((-c + kStoreRegionChunks - 1) / kStoreRegionChunks));
}

// One region file generation
struct RegionKey
{
    SpaceId space       = 0;
    int32_t rx          = 0;
    int32_t rz          = 0;
    uint32_t generation = 0;
    bool operator<(const RegionKey& o) const
    {
        return std::tie(space, rx, rz, generation) < std::tie(o.space, o.rx, o.rz, o.generation);
    }
    bool operator==(const RegionKey& o) const
    {
        return space == o.space && rx == o.rx && rz == o.rz && generation == o.generation;
    }
    bool sameRegion(const RegionKey& o) const { return space == o.space && rx == o.rx && rz == o.rz; }
    // regions/<space>/<rx>.<rz>.<g>.sgr
    std::string path() const
    {
        return "regions/" + std::to_string(space) + "/" + std::to_string(rx) + "." + std::to_string(rz) + "." +
               std::to_string(generation) + ".sgr";
    }
};
inline RegionKey regionKeyOf(const ChunkId& chunk, uint32_t generation)
{
    RegionKey k;
    k.space      = chunk.space;
    k.rx         = isSpaceRecordKey(chunk) ? 0 : regionOf(chunk.x);
    k.rz         = isSpaceRecordKey(chunk) ? 0 : regionOf(chunk.z);
    k.generation = generation;
    return k;
}

// Where a record lies
struct RecordLocation
{
    RegionKey file;
    uint64_t offset = 0;
    uint32_t length = 0;  // header + payload + checksum
    uint64_t commit = 0;
    bool operator==(const RecordLocation& o) const
    {
        return file == o.file && offset == o.offset && length == o.length && commit == o.commit;
    }
};

inline std::string worldFileName(uint64_t commit)
{
    return "world." + std::to_string(commit) + ".sgw";
}
inline const char* kDirHeaderName = "header.sgh";

// The FNV-1a 64 trailer of a file
inline void appendTrailer(std::vector<uint8_t>& out)
{
    const uint64_t h = fnv1a64Bytes(out.data(), out.size());
    for (int i = 0; i < 8; ++i) out.push_back(uint8_t(h >> (8 * i)));
}
inline bool checkTrailer(const std::vector<uint8_t>& bytes)
{
    if (bytes.size() < 8) return false;
    uint64_t stored = 0;
    for (int i = 0; i < 8; ++i) stored |= uint64_t(bytes[bytes.size() - 8 + size_t(i)]) << (8 * i);
    return fnv1a64Bytes(bytes.data(), bytes.size() - 8) == stored;
}

}  // namespace Engine
}  // namespace Saiga
