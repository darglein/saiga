/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainBrush (docs/engine_design.md §12.2; S22): a terraforming brush of World::editTerrain (a nested visit struct,
// so game components can hold one), the engine's safety bounds and validateBrush, which checks a brush against a space
// before anything is quantized (BrushKernel quantizes and evaluates it with integer arithmetic only).

#pragma once

#include "saiga/core/math/Types.h"
#include "saiga/engine/ecs/Visit.h"
#include "saiga/engine/world/SpaceTemplate.h"
#include "saiga/export.h"

#include <cstdint>
#include <string>

namespace Saiga
{
namespace Engine
{
struct TerrainBrush
{
    static constexpr const char* name = "TerrainBrush";
    static constexpr uint32_t version = 1;

    // Count: visit's EnumNames convention
    enum Op : uint8_t
    {
        Add,
        Subtract,
        Flatten,
        Smooth,
        Count
    };
    Op op = Add;
    // x, z: the brush center (metres); y: the Flatten target (finite for every op, used only by Flatten)
    Vec3 center = Vec3::Zero();
    // Metres, (0, kMaxBrushRadius]
    double radius = 1;
    // Add/Subtract: metres at the center, [0, kMaxBrushStep]; Flatten/Smooth: the blend factor at the center, [0, 1]
    double strength = 0;

    static constexpr const char* opNames[] = {"add", "subtract", "flatten", "smooth"};

    template <class V>
    void visit(V& v)
    {
        v("op", op, EnumNames{opNames});
        v("center", center);
        v("radius", radius);
        v("strength", strength);
    }
};

// Metres: a brush touches at most 2 x 2 chunks (R-MECH-1: engine safety bounds, not gameplay limits)
constexpr double kMaxBrushRadius = 128;
// Metres per edit (Add/Subtract)
constexpr double kMaxBrushStep = 64;

// The checks of a brush on the doubles, before anything is quantized: finite x, y, z, radius and strength; op < Count;
// radius in (0, kMaxBrushRadius]; strength in [0, kMaxBrushStep] (Add/Subtract) or [0, 1] (Flatten/Smooth); x and z in
// [-kMaxBrushRadius, sizeChunks * 256 + kMaxBrushRadius] of the space; for Flatten |y| <= kTerrainHeightLimit *
// kHeightUnit (8192 m). False with an error naming the field and the bound
SAIGA_ENGINE_API bool validateBrush(const TerrainBrush& brush, const SpaceTemplate& space, std::string* error);

}  // namespace Engine
}  // namespace Saiga
