/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/VisitBinary.h"
#include "saiga/engine/world/SpaceTemplate.h"

#include <cstdint>
#include <string>
#include <vector>

// The save blob of World::save (docs/engine_design.md §15.1, S11): the magic "SGWS", the format version, then sections
// (a 4-character tag and a byte length) in the fixed order of kSaveSections. Little-endian, the binary visit form for
// values. An unknown section, one out of order or twice, and a missing required section are load errors. Optional
// sections (S21, D7) may be absent; each starts with its own format version (u32), and a version above the entry's
// maxVersion is a load error naming the section (kSaveFormatVersion stays 1: older builds reject a blob with an
// optional section as an unexpected section).
//
//   HEAD  engine version (string), steps per second, step index, time
//   SCHM  the schema table (SchemaTable::write)
//   SPCE? (S21, version 1; only with more than one space) the space count (u32), then per SpaceId the template name
//         (u32 length + bytes, at most 256) and the kind (u8, SpaceKind); names may repeat (S23a)
//   GENR? (S23a, version 1; only when the world writes generation state: a non-legacy generator or chunk records) the
//         seed (u64); the template count (u32), per template the generator name (string, u64 length + bytes); the
//         generator count (u32), per generator in name order: name (string), pinned version (u16), current version
//         (u16), slot count (u16), per slot the name (string) and the flags (u32); the space count N (u32) and the
//         combined plan hash (u64) of the spaces 0..N-1
//   TERR? (S22, version 1; only with a chunk with terrain state, sequence > 0) the space count (u32); per space with
//         such chunks in SpaceId order: SpaceId (u16), base spacing (u8), N (u16), chunk count (u32); per chunk in
//         ChunkId order: x (i32), z (i32), sequence (u32, > 0), flags (u8: bit 0 = deltas present); with deltas the
//         encoded length (u64) and the dense N x N delta (TerrainDeltaCodec, zstd)
//   CREC? (S23a, version 1; only with chunk records) the count of spaces with records (u32); per such space in SpaceId
//         order: SpaceId (u16), record count (u32); per record in ChunkId order: x (i32), z (i32), placement version
//         (u16), stamped (u8: 0 or 1, the host's stamped set)
//   ENTS  slot count; per slot from 1: generation (u32), state (u8: 0 free, 2 alive), flags (u8); the free list
//   COMP  per saved component type: schema index (u32), count (u64), byte length (u64), then per component the entity
//         index (u32) and the binary form, in the pool's dense order (Transform: index order)
//   EVNT  per event type with pending events: schema index, count, byte length, the events in emit order
//   ACTN  the queued actions (peer u8, sequence u64, player Entity, action u32, aim Vec3), the action sequence
//   STAT  the RNG state (state, increment), the Persistent counter

namespace Saiga
{
namespace Engine
{
constexpr uint32_t kSaveFormatVersion = 1;
constexpr uint8_t kSaveMagic[4]       = {'S', 'G', 'W', 'S'};

constexpr uint32_t saveTag(const char (&t)[5])
{
    return uint32_t(uint8_t(t[0])) | uint32_t(uint8_t(t[1])) << 8 | uint32_t(uint8_t(t[2])) << 16 |
           uint32_t(uint8_t(t[3])) << 24;
}
constexpr uint32_t kTagHeader       = saveTag("HEAD");
constexpr uint32_t kTagSchema       = saveTag("SCHM");
constexpr uint32_t kTagSpaces       = saveTag("SPCE");
constexpr uint32_t kTagGeneration   = saveTag("GENR");
constexpr uint32_t kTagTerrain      = saveTag("TERR");
constexpr uint32_t kTagChunkRecords = saveTag("CREC");
constexpr uint32_t kTagEntities     = saveTag("ENTS");
constexpr uint32_t kTagComponents   = saveTag("COMP");
constexpr uint32_t kTagEvents       = saveTag("EVNT");
constexpr uint32_t kTagActions      = saveTag("ACTN");
constexpr uint32_t kTagState        = saveTag("STAT");

// The section order (D7): HEAD, SCHM, SPCE?, GENR?, TERR?, CREC?, ENTS, COMP, EVNT, ACTN, STAT (? optional). S22
// inserts TERR, S23a GENR and CREC at their positions (the recipe of the S21 plan, Scope 5). maxVersion: the highest
// format version of an optional section this build reads (0 for the required ones)
struct SaveSectionEntry
{
    uint32_t tag;
    bool optional;
    uint32_t maxVersion;
};
constexpr SaveSectionEntry kSaveSections[] = {
    {kTagHeader, false, 0}, {kTagSchema, false, 0},      {kTagSpaces, true, 1},    {kTagGeneration, true, 1},
    {kTagTerrain, true, 1}, {kTagChunkRecords, true, 1}, {kTagEntities, false, 0}, {kTagComponents, false, 0},
    {kTagEvents, false, 0}, {kTagActions, false, 0},     {kTagState, false, 0},
};

// SPCE: the template name and kind per SpaceId
struct SpaceTable
{
    struct Row
    {
        std::string name;
        SpaceKind kind = SpaceKind::OpenWorld;
    };
    std::vector<Row> rows;
};
constexpr uint32_t kSpaceTableVersion = 1;

// TERR (S22): the terrain modification state per space; dense deltas decoded (N x N, row-major, row = z)
struct TerrainSection
{
    struct Chunk
    {
        int32_t x = 0, z = 0;
        uint32_t sequence = 0;
        bool hasDeltas    = false;
        std::vector<int32_t> deltas;  // N * N with deltas, empty without
    };
    struct Space
    {
        SpaceId space   = 0;
        uint8_t spacing = 0;
        uint16_t N      = 0;
        std::vector<Chunk> chunks;
    };
    std::vector<Space> spaces;
};
constexpr uint32_t kTerrainSectionVersion = 1;

// GENR (S23a): the generation state of the save
struct GenerationSection
{
    uint64_t seed = 0;
    std::vector<std::string> templateGenerators;  // per template, the generator name
    struct Slot
    {
        std::string name;
        uint32_t flags = 0;
    };
    struct Generator
    {
        std::string name;
        uint16_t pinned = 0, current = 0;
        std::vector<Slot> slots;
    };
    std::vector<Generator> generators;  // name order
    uint32_t spaceCount   = 0;
    uint64_t combinedHash = 0;
};
constexpr uint32_t kGenerationSectionVersion = 1;

// CREC (S23a): the chunk records
struct ChunkRecordSection
{
    struct Record
    {
        int32_t x = 0, z = 0;
        uint16_t placementVersion = 0;
        bool stamped              = false;
    };
    struct Space
    {
        SpaceId space = 0;
        std::vector<Record> records;
    };
    std::vector<Space> spaces;
};
constexpr uint32_t kChunkRecordSectionVersion = 1;

inline std::string tagName(uint32_t tag)
{
    std::string s;
    for (int i = 0; i < 4; ++i)
    {
        const char c = char((tag >> (8 * i)) & 0xFF);
        s += (c >= 32 && c < 127) ? c : '?';
    }
    return s;
}

// Overwrites a u64 written earlier at position at (a length known only afterwards)
inline void patchU64(std::vector<uint8_t>& out, size_t at, uint64_t value)
{
    for (int i = 0; i < 8; ++i) out[at + size_t(i)] = uint8_t(value >> (8 * i));
}

// Writes one section: begin() remembers the length field, end() fills it in
class SectionWriter
{
   public:
    explicit SectionWriter(std::vector<uint8_t>& out) : out(out), w(out) {}
    BinaryWriter& writer() { return w; }
    void begin(uint32_t tag)
    {
        w.write(tag);
        lengthAt = out.size();
        w.write(uint64_t(0));
    }
    // An optional section: the tag, the length placeholder and the section's own format version
    void beginOptional(uint32_t tag, uint32_t version)
    {
        begin(tag);
        w.write(version);
    }
    void end() { patchU64(out, lengthAt, uint64_t(out.size() - lengthAt - 8)); }

   private:
    std::vector<uint8_t>& out;
    BinaryWriter w;
    size_t lengthAt = 0;
};

}  // namespace Engine
}  // namespace Saiga
