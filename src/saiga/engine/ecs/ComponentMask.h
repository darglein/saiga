/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Ids.h"

#include <array>
#include <cstdint>

namespace Saiga
{
namespace Engine
{
// One bit per component ID (256 bits; docs/engine_design.md §7.1)
class ComponentMask
{
   public:
    void set(ComponentId id) { bits[id >> 6] |= uint64_t(1) << (id & 63); }
    void reset(ComponentId id) { bits[id >> 6] &= ~(uint64_t(1) << (id & 63)); }
    bool test(ComponentId id) const { return (bits[id >> 6] >> (id & 63)) & 1; }

    // Every bit of other is set in this mask
    bool containsAll(const ComponentMask& other) const
    {
        for (size_t i = 0; i < bits.size(); ++i)
            if ((bits[i] & other.bits[i]) != other.bits[i]) return false;
        return true;
    }
    bool any() const { return (bits[0] | bits[1] | bits[2] | bits[3]) != 0; }
    void clear() { bits = {}; }

    bool operator==(const ComponentMask& o) const { return bits == o.bits; }
    bool operator!=(const ComponentMask& o) const { return bits != o.bits; }

    const std::array<uint64_t, 4>& words() const { return bits; }

   private:
    std::array<uint64_t, 4> bits = {};
};

}  // namespace Engine
}  // namespace Saiga
