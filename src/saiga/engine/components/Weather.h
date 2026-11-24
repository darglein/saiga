/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/Types.h"
#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// The tile of the cloud coverage (m): cloudOffset wraps into [0, kCloudTile) per axis, a multiple of the coverage
// texture's period, so the wrap never jumps (§10.7)
constexpr double kCloudTile = 16384;

// The weather of a space (docs/engine_design.md §7.8, §10.7; S36): space state when WorldSetup::environment.enabled,
// else a plain component no entity has. How weather changes is game policy (R-MECH-1): the optional worldTimeSystem
// only moves cloudOffset with the wind and normalises the values. Written by setWeather (tools, the Lighting tab).
// Replicate::Host, saved. rain and snowCover are state only until S43.
struct Weather
{
    static constexpr const char* name = "Weather";
    static constexpr ComponentId id   = ComponentId(EngineComponent::Weather);
    static constexpr uint32_t version = 1;

    double cloudiness    = 0.3;      // [0, 1]
    double rain          = 0;        // [0, 1]
    double fogDensity    = 0.00015;  // 1/m, [0, 0.05]
    double snowCover     = 0;        // [0, 1]
    double waveAmplitude = 1;
    double waveSteepness = 0.5;
    Vec2 wind            = Vec2(3, 1);    // m/s in x, z
    Vec2 cloudOffset     = Vec2::Zero();  // m, accumulated from the wind, wrapped to [0, kCloudTile)
    double validAt       = 0;             // World::time() at which cloudOffset is valid (as WorldTime::validAt)

    template <class V>
    void visit(V& v)
    {
        v("cloudiness", cloudiness);
        v("rain", rain);
        v("fogDensity", fogDensity);
        v("snowCover", snowCover);
        v("waveAmplitude", waveAmplitude);
        v("waveSteepness", waveSteepness);
        v("wind", wind);
        v("cloudOffset", cloudOffset);
        v("validAt", validAt);
    }
};

}  // namespace Engine
}  // namespace Saiga
