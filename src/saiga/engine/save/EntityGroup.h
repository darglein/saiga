/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The entity group form (docs/engine_design.md §9.6, §15.1; S25, format 1): a root and its descendants as stored by an
// unload. Untrusted input, bounded like a save:
//   unload step (u64; the elapsed time at reload is (step - unloadStep) * dt)
//   entity count n (u32, 1..65536): local entity 0 is the root, then the descendants, parents before children
//   the PersistentIds of the group (count u32, then u64 each, ascending): checked at loads without decoding
//   the component block count (u32), then the blocks in component ID order (Transform first), each as in COMP: schema
//   index (u32), value count (u32), byte length (u64), per value the slot (u32, local index + 1, as in the handles) and
//   the binary form
// Entity fields inside the group are written as Entity{local + 1, 0}, every other Entity value (outside the group,
// dead, null) as Entity{}. The schema table number is stored next to the bytes (StoredGroup, RGRP, EGRP), never inside
// them.

#pragma once

#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/save/SaveReader.h"
#include "saiga/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

constexpr uint32_t kMaxGroupEntities = 65536;

// The structure of a group (blocks index into the group bytes)
struct GroupForm
{
    uint64_t unloadStep = 0;
    uint32_t entities   = 0;
    std::vector<uint64_t> persistentIds;
    std::vector<ParsedSave::Block> blocks;
};

// Checks the structure (counts and lengths within bounds, schema indices below schemaTypes, ascending unique non-zero
// PersistentIds, no bytes left over); the values are decoded at reload (GroupLoader)
SAIGA_ENGINE_API bool parseGroupForm(const uint8_t* data, size_t size, size_t schemaTypes, GroupForm& out,
                                     std::string& error);

// The result of writing a group
struct GroupWriteResult
{
    std::vector<uint8_t> bytes;
    std::vector<Entity> entities;  // local order
    uint64_t droppedReferences = 0;
};

// Writes the group of root (its subtree in collectSubtree order) with the saved components of the current schema
// table schema (SchemaTable::fromWorld). Main thread, between steps or at a flush
SAIGA_ENGINE_API void writeGroup(const World& world, Entity root, const SchemaTable& schema, uint64_t step,
                                 GroupWriteResult& out);

// The class name of the root (its Behaviour's class ID in the group's schema table; "" without a game object) and the
// entity count, for diffs and dumps; false if the group is malformed
SAIGA_ENGINE_API bool describeGroup(const std::vector<uint8_t>& bytes, const SchemaTable& schema,
                                    std::string& rootClass, uint32_t& entities);

}  // namespace Engine
}  // namespace Saiga
