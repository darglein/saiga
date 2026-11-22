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
    std::optional<SpaceTable> spaces;  // SPCE
};

SAIGA_ENGINE_API bool parseSave(const std::vector<uint8_t>& data, ParsedSave& out, std::string& error);
// SPCE format 1 (the one signature of the optional-section parsers: parseX(BinaryReader&, version, X&, error)): at
// most 65535 rows, names of at most 256 bytes, a known kind; no bytes left over
SAIGA_ENGINE_API bool parseSpaceTable(BinaryReader& r, uint32_t version, SpaceTable& out, std::string& error);

}  // namespace Engine
}  // namespace Saiga
