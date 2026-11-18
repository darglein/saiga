/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// A handle to an entity (docs/engine_design.md §7.1): the slot index and the generation of the slot when the entity was
// created. A destroyed entity's handle is stale (the slot's generation moved on); {0, 0} is invalid and never alive.
struct Entity
{
    uint32_t index      = 0;
    uint32_t generation = 0;

    bool valid() const { return index != 0; }
    bool operator==(const Entity& o) const { return index == o.index && generation == o.generation; }
    bool operator!=(const Entity& o) const { return !(*this == o); }
    bool operator<(const Entity& o) const { return index < o.index || (index == o.index && generation < o.generation); }
};

}  // namespace Engine
}  // namespace Saiga
