/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/save/SchemaTable.h"
#include "saiga/export.h"

#include <cstdint>
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
};

SAIGA_ENGINE_API bool parseSave(const std::vector<uint8_t>& data, ParsedSave& out, std::string& error);

}  // namespace Engine
}  // namespace Saiga
