/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// One difference between two saves (S11)
struct SAIGA_ENGINE_API SaveDifference
{
    // field, only_in_a, only_in_b (also a reused slot: both), component_only_in_a, component_only_in_b, flags, step,
    // time, rng, events, actions, free_list, counter
    std::string kind;
    Entity entity;                 // entity differences (in a's numbering; the index is the same in both)
    std::string component, field;  // field differences
    json11::Json a, b;             // the values (JSON form of VisitJson.h)
    std::string className;         // only_in_a / only_in_b: the class of the entity, if it has one
    json11::Json position;         // only_in_a / only_in_b: its position
    json11::Json toJson() const;
    std::string describe() const;  // "entity 12:3 Health.hp: 10 != 7"
};

struct DiffFilter
{
    bool hasEntity            = false;
    uint32_t entityIndex      = 0;      // only this entity index (and no world-level differences)
    bool hasGeneration        = false;  // and only with this generation (a reused slot is another entity)
    uint32_t entityGeneration = 0;
    std::vector<std::string> components;  // only these components (empty: all)
    size_t limit = 0;                     // at most this many differences in the result (0: all)
    size_t after = 0;                     // skip the first differences (paging)
};

struct SaveDiff
{
    std::string error;  // a save could not be read
    std::vector<SaveDifference> differences;
    size_t total = 0;      // all differences that pass the filter
    json11::Json summary;  // counts per kind and per component
};

// Compares two saves by name (they may come from different builds): world-level state first, then per entity in index
// order, components by name, fields by name. Saved state only.
SAIGA_ENGINE_API SaveDiff diffSaves(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                                    const DiffFilter& filter = {});

}  // namespace Engine
}  // namespace Saiga
