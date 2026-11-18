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
// A reference to an entity that survives unloading and saving (docs/engine_design.md §9.6). The value type only;
// world.makePersistent and world.resolve come in S10. 0 is invalid.
struct PersistentId
{
    uint64_t value = 0;

    bool valid() const { return value != 0; }
    bool operator==(const PersistentId& o) const { return value == o.value; }
    bool operator!=(const PersistentId& o) const { return value != o.value; }
    bool operator<(const PersistentId& o) const { return value < o.value; }
};

}  // namespace Engine
}  // namespace Saiga
