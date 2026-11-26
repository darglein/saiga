/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/save/SaveFormat.h"
#include "saiga/engine/save/SchemaTable.h"
#include "saiga/export.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
/**
 * The sections of a save blob, checked for structure (lengths, counts, known tags, the slot table) but not decoded per
 * value: World::tryLoad decodes the components by the schema, diffSaves compares them (S11).
 */
struct ParsedSave
{
    std::string engineVersion;
    double stepsPerSecond = 0;
    uint64_t steps        = 0;
    double time           = 0;
    SchemaTable schema;

    struct Slot
    {
        uint32_t generation = 0;
        uint8_t state       = 0;  // SlotState: 0 free, 2 alive
        uint8_t flags       = 0;
    };
    std::vector<Slot> slots;  // slots[0] is the unused slot
    std::vector<uint32_t> freeList;

    // The values of one type: count values in [begin, end) of the blob (components: each an entity index, then the
    // value)
    struct Block
    {
        uint32_t schemaIndex = 0;
        uint64_t count       = 0;
        size_t begin = 0, end = 0;
    };
    std::vector<Block> components, events;

    struct Action
    {
        uint8_t peer      = 0;
        uint64_t sequence = 0;
        Entity player;
        uint32_t action = 0;
        Vec3 aim        = Vec3::Zero();
    };
    std::vector<Action> actions;
    uint64_t actionSequence = 0;
    uint64_t rngState = 0, rngIncrement = 0;
    uint64_t persistentCounter = 0;

    // ---- the optional sections (S21, D7): one member each, absent when the section is not in the blob
    // A large optional section kept as its raw bytes [begin, end) of the blob after its version (decoded by the
    // loader's own pass before the commit; S22's TERR)
    struct Range
    {
        uint32_t version = 0;
        size_t begin = 0, end = 0;
    };
    std::optional<SpaceTable> spaces;                // SPCE
    std::optional<Range> terrain;                    // TERR (parseTerrainSection)
    std::optional<GenerationSection> generation;     // GENR (S23a)
    std::optional<ChunkRecordSection> chunkRecords;  // CREC (S23a)
    std::optional<Range> storedGroups;               // EGRP (S25; parseStoredGroups)
};

// EGRP format 1 (S25): the extra schema tables (number 0 is the blob's SCHM), the chunks with groups and the unloaded
// spaces. Structure only (sorted unique chunks and spaces, table numbers in range, group forms structurally valid); the
// loader checks the rest against the world
struct StoredGroupsSection
{
    std::vector<SchemaTable> extraTables;
    struct Group
    {
        uint32_t table = 0;
        std::vector<uint8_t> bytes;
    };
    struct Chunk
    {
        ChunkId chunk;
        std::vector<Group> groups;
    };
    std::vector<Chunk> chunks;
    struct Space
    {
        SpaceId space = 0;
        bool hasGroup = false;
        Group group;
    };
    std::vector<Space> spaces;
};
constexpr uint32_t kStoredGroupsSectionVersion = 1;
// The section's bytes after its version (ParsedSave::storedGroups' range); schema: the blob's SCHM
SAIGA_ENGINE_API bool parseStoredGroups(const uint8_t* data, size_t size, uint32_t version, const SchemaTable& schema,
                                        StoredGroupsSection& out, std::string& error);

SAIGA_ENGINE_API bool parseSave(const std::vector<uint8_t>& data, ParsedSave& out, std::string& error);
// SPCE format 1 (the one signature of the optional-section parsers: parseX(BinaryReader&, version, X&, error)): at
// most 65535 rows, names of at most 256 bytes, a known kind; no bytes left over
SAIGA_ENGINE_API bool parseSpaceTable(BinaryReader& r, uint32_t version, SpaceTable& out, std::string& error);
// GENR format 1 (S23a): at most 65535 templates, generators and slots per generator; no bytes left over. The loader
// checks it against the setup (seed, names, versions, slot tables, plans)
SAIGA_ENGINE_API bool parseGenerationSection(BinaryReader& r, uint32_t version, GenerationSection& out,
                                             std::string& error);
// CREC format 1 (S23a) or 2 (S23b: tombstones): ascending unique spaces and chunks (ChunkId order), stamped 0 or 1; no
// bytes left over. The loader checks the spaces, the chunk ranges, the versions and the keys against the plans
SAIGA_ENGINE_API bool parseChunkRecordSection(BinaryReader& r, uint32_t version, ChunkRecordSection& out,
                                              std::string& error);
// TERR format 1 (S22): the structure and the codec's bounds, checked without a world: spacing 1 or 2 with its N,
// ascending unique spaces and chunks (ChunkId order), sequence > 0, known flags, the dense delta decoded (N * N values,
// |delta| <= 2^24, no bytes left over), deltas present only with a non-zero value; no bytes left over. The loader
// checks the rest against the world (the spaces, the chunk range, the border rule)
// The section's bytes after its version (ParsedSave::terrain's range)
SAIGA_ENGINE_API bool parseTerrainSection(const uint8_t* data, size_t size, uint32_t version, TerrainSection& out,
                                          std::string& error);

}  // namespace Engine
}  // namespace Saiga
