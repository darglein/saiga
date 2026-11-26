/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The encoders and decoders of the save directory's files (docs/engine_design.md §15.1; S25; layout in
// SaveDirectoryFormat.h): records with their sections (RTER, RREC, RGRP, RSPC), the chunk index (CIDX), the header and
// the world-state file. Decoders take untrusted input: every count, length and version is checked in every build and an
// error names the chunk and the section.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/save/SaveDirectoryFormat.h"
#include "saiga/engine/save/SchemaTable.h"
#include "saiga/export.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// zstd with the bound check of untrusted input (the frame header's content size, at most kMaxUncompressedPart)
SAIGA_ENGINE_API void zstdCompress(const uint8_t* data, size_t size, int level, std::vector<uint8_t>& out);
SAIGA_ENGINE_API bool zstdDecompress(const uint8_t* data, size_t size, std::vector<uint8_t>& out, std::string& error);

using GroupBytes = std::shared_ptr<const std::vector<uint8_t>>;

// One stored entity group inside a record or the index: its schema table number and its bytes (the group form)
struct RecordGroup
{
    uint32_t schema = 0;
    GroupBytes bytes;
};

// The complete state of a chunk (kind 0) or of an unloaded space (kind 1) at one commit
struct RecordContent
{
    uint8_t kind = kRecordKindChunk;
    ChunkId key;  // the chunk, or spaceRecordKey(space)
    uint64_t commit = 0;
    // RTER (a chunk with terrain state, sequence > 0): the dense N x N delta of the chunk's closed range with deltas
    bool hasTerrain   = false;
    uint32_t sequence = 0;
    bool hasDeltas    = false;
    uint16_t N        = 0;
    std::vector<int32_t> deltas;
    // RREC (a chunk with a chunk record)
    bool hasRecord            = false;
    uint16_t placementVersion = 0;
    bool stamped              = false;
    std::vector<InstanceKey> tombstones;
    // RGRP: the groups in stored order (oldest first)
    std::vector<RecordGroup> groups;
    // RSPC (kind 1): the space-state entity as one group (absent: the world has no space-state components)
    bool hasSpaceGroup = false;
    RecordGroup spaceGroup;
};

// The whole record (header, zstd payload, checksum); false (nothing usable in out) when the uncompressed payload
// exceeds kMaxUncompressedPart, which the decoder would refuse: such a record must not be committed
SAIGA_ENGINE_API bool encodeRecord(const RecordContent& content, int zstdLevel, std::vector<uint8_t>& out);
// A whole record: the checksum, the header (magic, format, kind), the payload's frame and sections
SAIGA_ENGINE_API bool decodeRecord(const uint8_t* data, size_t size, RecordContent& out, std::string& error);
// Only the header of a record (the open compares it with its index entry)
struct RecordHeader
{
    uint8_t kind = 0;
    ChunkId key;
    uint64_t commit        = 0;
    uint32_t payloadLength = 0;
};
SAIGA_ENGINE_API bool decodeRecordHeader(const uint8_t* data, size_t size, RecordHeader& out, std::string& error);

// The index summary of a chunk (enough to load it without its record when it has no deltas, no tombstones and no
// groups: the records of every visited chunk, Scope 12)
struct IndexSummary
{
    uint32_t sequence         = 0;  // 0: no terrain state
    bool deltas               = false;
    bool record               = false;
    bool stamped              = false;
    uint64_t deltaHash        = 0;
    uint16_t placementVersion = 0;
    uint32_t tombstones       = 0;
    struct Group
    {
        uint32_t schema = 0;
        uint64_t hash   = 0;
    };
    std::vector<Group> groups;
    // The record is needed to load the chunk's state
    bool needsRecord() const { return deltas || tombstones > 0; }
    bool empty() const { return sequence == 0 && !record && !stamped && groups.empty(); }
};
struct IndexEntry
{
    uint8_t kind = kRecordKindChunk;
    ChunkId key;
    std::optional<RecordLocation> location;  // none: the summary is the whole state
    IndexSummary summary;
};
struct ChunkIndex
{
    // Every region file a location names, with its committed end offset
    std::vector<std::pair<RegionKey, uint64_t>> files;
    // In (kind, space, chunk) order
    std::vector<IndexEntry> entries;
};
SAIGA_ENGINE_API void encodeIndex(const ChunkIndex& index, std::vector<uint8_t>& out);
// Sorted and unique entries, files and offsets within the table and below each file's committed end
SAIGA_ENGINE_API bool decodeIndex(const uint8_t* data, size_t size, ChunkIndex& out, std::string& error);

// The header file
SAIGA_ENGINE_API void encodeDirHeader(const std::string& engineVersion,
                                      const std::vector<std::shared_ptr<const SchemaTable>>& tables,
                                      std::vector<uint8_t>& out);
SAIGA_ENGINE_API bool decodeDirHeader(const std::vector<uint8_t>& bytes, std::string& engineVersion,
                                      std::vector<std::shared_ptr<const SchemaTable>>& tables, std::string& error);

// The world-state file
struct WorldFile
{
    uint64_t commit       = 0;
    uint32_t tablesNeeded = 0;
    std::vector<uint8_t> blob;   // the world-state blob
    std::vector<uint8_t> index;  // the encoded CIDX
};
// false when a part exceeds kMaxUncompressedPart (as encodeRecord)
SAIGA_ENGINE_API bool encodeWorldFile(const WorldFile& file, int zstdLevel, std::vector<uint8_t>& out);
// false: corruption (checksum, framing); the caller tries the next candidate
SAIGA_ENGINE_API bool decodeWorldFile(const std::vector<uint8_t>& bytes, WorldFile& out, std::string& error);

// The 24-byte region file header
SAIGA_ENGINE_API std::vector<uint8_t> encodeRegionHeader(const RegionKey& key);
SAIGA_ENGINE_API bool checkRegionHeader(const uint8_t* data, size_t size, const RegionKey& key, std::string& error);

}  // namespace Engine
}  // namespace Saiga
