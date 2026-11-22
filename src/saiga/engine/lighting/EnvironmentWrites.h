/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The validated writes of the time of day and the weather (docs/engine_design.md §10.7; S36): the Lighting tab and the
// tools (set_time_of_day, set_weather) write through them; and the normalisation of both components (worldTimeSystem,
// the environment extract). Visit has no value limits, so a raw set_global can write anything: the normalisation
// repairs it (wrap, clamp, a non-finite value -> the default).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/Types.h"
#include "saiga/engine/components/Weather.h"
#include "saiga/engine/components/WorldTime.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/export.h"

#include <cstdint>
#include <optional>
#include <string>

namespace Saiga
{
namespace Engine
{
class World;

// The ranges (A1 of the S36 plan)
constexpr double kMaxTimeRate      = 3600;  // game hours per second, |rate|
constexpr double kMaxFogDensity    = 0.05;
constexpr double kMaxWind          = 200;  // m/s per axis
constexpr double kMaxWaveAmplitude = 100;

// Wraps timeOfDay into [0, 24) and counts the days; clamps rate; a non-finite value -> its default
SAIGA_ENGINE_API void normaliseWorldTime(WorldTime& t);
// Clamps the ranges, wraps cloudOffset into [0, kCloudTile); a non-finite value -> its default
SAIGA_ENGINE_API void normaliseWeather(Weather& w);
// x wrapped into [0, period)
SAIGA_ENGINE_API double wrapPeriod(double x, double period);

// The fields to change (unset: unchanged)
struct WorldTimeChange
{
    std::optional<double> hours;  // [0, 24)
    std::optional<int64_t> day;
    std::optional<double> rate;  // [-3600, 3600]
};
struct WeatherChange
{
    std::optional<double> cloudiness, rain, fogDensity, snowCover, waveAmplitude, waveSteepness;
    std::optional<Vec2> wind, cloudOffset;
};

// Validates every field (finite, in range: the error names the field and the range), then writes the world's WorldTime
// (validAt = World::time()) and logs the change under "api". Between steps only; refused on a client (the host writes
// it, R-NET-2) and in a world without WorldTime (EnvironmentSetup::enabled). previous: the values before the write
SAIGA_ENGINE_API bool setWorldTime(World& world, const WorldTimeChange& change, std::string& error,
                                   WorldTime* previous = nullptr);
// The same for the Weather of a space
SAIGA_ENGINE_API bool setWeather(World& world, SpaceId space, const WeatherChange& change, std::string& error,
                                 Weather* previous = nullptr);

// The world has WorldTime / Weather as world and space state (EnvironmentSetup::enabled)
SAIGA_ENGINE_API bool hasWorldTime(const World& world);
SAIGA_ENGINE_API bool hasWeather(const World& world);

}  // namespace Engine
}  // namespace Saiga
