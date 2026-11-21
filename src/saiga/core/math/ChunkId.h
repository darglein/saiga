/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

#include <cmath>
#include <cstdint>
#include <tuple>

// Value types of the world layout (docs/engine_design.md §4 glossary, §12): spaces, chunks, regions and the keys of
// generated instances. In saiga_core so that saiga_vulkan can use them without the engine. Each has name, version and
// visit like a nested component struct (§7.2), so they can be fields of components.

namespace Saiga
{
// An independent world (overworld = 0, dungeons, ...)
using SpaceId = uint16_t;

// 256 m x 256 m square of a space; origin (x * 256, 0, z * 256)
struct ChunkId
{
    static constexpr const char* name = "ChunkId";
    static constexpr uint32_t version = 1;

    SpaceId space = 0;
    int32_t x = 0, z = 0;

    template <class V>
    void visit(V& v)
    {
        v("space", space);
        v("x", x);
        v("z", z);
    }

    bool operator==(const ChunkId& o) const { return space == o.space && x == o.x && z == o.z; }
    bool operator!=(const ChunkId& o) const { return !(*this == o); }
    bool operator<(const ChunkId& o) const { return std::tie(space, x, z) < std::tie(o.space, o.x, o.z); }
};

// A square of 2^level x 2^level chunks (far content, terrain levels)
struct RegionId
{
    static constexpr const char* name = "RegionId";
    static constexpr uint32_t version = 1;

    SpaceId space = 0;
    uint8_t level = 0;
    int32_t x = 0, z = 0;

    template <class V>
    void visit(V& v)
    {
        v("space", space);
        v("level", level);
        v("x", x);
        v("z", z);
    }

    bool operator==(const RegionId& o) const { return space == o.space && level == o.level && x == o.x && z == o.z; }
    bool operator!=(const RegionId& o) const { return !(*this == o); }
    bool operator<(const RegionId& o) const
    {
        return std::tie(space, level, x, z) < std::tie(o.space, o.level, o.x, o.z);
    }
};

// The key of a generated instance in its chunk: generator slot and the counter of that generator (§12.1)
struct InstanceKey
{
    static constexpr const char* name = "InstanceKey";
    static constexpr uint32_t version = 1;

    uint32_t generatorSlot = 0, counter = 0;

    template <class V>
    void visit(V& v)
    {
        v("generator_slot", generatorSlot);
        v("counter", counter);
    }

    bool operator==(const InstanceKey& o) const { return generatorSlot == o.generatorSlot && counter == o.counter; }
    bool operator!=(const InstanceKey& o) const { return !(*this == o); }
    bool operator<(const InstanceKey& o) const
    {
        return std::tie(generatorSlot, counter) < std::tie(o.generatorSlot, o.counter);
    }
};

// The size of a chunk in meters
constexpr double chunkSize = 256;

// The origin of the chunk in its space (double, R-MATH-1): (x, 0, z) * 256 m
inline Vec3 chunkOrigin(const ChunkId& chunk)
{
    return Vec3(chunk.x * chunkSize, 0, chunk.z * chunkSize);
}

// The origin of the region in its space: (x, 0, z) * 256 m * 2^level (x, z in region units of 2^level chunks, §12)
inline Vec3 regionOrigin(const RegionId& region)
{
    const double size = std::ldexp(chunkSize, region.level);
    return Vec3(region.x * size, 0, region.z * size);
}

}  // namespace Saiga
