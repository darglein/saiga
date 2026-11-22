/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The validated writes of WorldTime and Weather and their normalisation. Writes WorldTime (global) and Weather (space
// state).

#include "saiga/engine/lighting/EnvironmentWrites.h"

#include "saiga/engine/core/Log.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Saiga
{
namespace Engine
{
namespace
{
double finiteOr(double v, double fallback)
{
    return std::isfinite(v) ? v : fallback;
}
double clampFinite(double v, double lo, double hi, double fallback)
{
    return std::clamp(finiteOr(v, fallback), lo, hi);
}
bool checkRange(const char* name, double v, double lo, double hi, bool hiOpen, std::string& error)
{
    if (std::isfinite(v) && v >= lo && (hiOpen ? v < hi : v <= hi)) return true;
    error = std::string(name) + ": expected a finite number in [" + std::to_string(lo) + ", " + std::to_string(hi) +
            (hiOpen ? ")" : "]");
    return false;
}
// |timeOfDay| at or beyond this many days is not a time but garbage (a raw set_global): it is reset, which keeps the
// day count's conversion to int64 defined (the range of set_time_of_day's day)
constexpr double kMaxDays = 1e15;

// day + days, saturated at the int64 limits (no signed overflow)
int64_t addDays(int64_t day, int64_t days)
{
    if (days > 0 && day > std::numeric_limits<int64_t>::max() - days) return std::numeric_limits<int64_t>::max();
    if (days < 0 && day < std::numeric_limits<int64_t>::min() - days) return std::numeric_limits<int64_t>::min();
    return day + days;
}

std::string refusal(const World& world, const char* component)
{
    if (world.inStep()) return std::string(component) + ": only between steps";
    if (!world.isHost())
        return std::string(component) +
               ": the entity is replicated (a client writes only the owner-authoritative components of the entities "
               "it owns)";
    return {};
}
}  // namespace

double wrapPeriod(double x, double period)
{
    double r = std::fmod(x, period);
    if (r < 0) r += period;
    // fmod of a tiny negative value can round up to the period
    if (r >= period) r = 0;
    return r;
}

void normaliseWorldTime(WorldTime& t)
{
    const WorldTime defaults;
    t.rate    = clampFinite(t.rate, -kMaxTimeRate, kMaxTimeRate, defaults.rate);
    t.validAt = finiteOr(t.validAt, defaults.validAt);
    if (!std::isfinite(t.timeOfDay) || !(std::abs(t.timeOfDay) < 24 * kMaxDays)) t.timeOfDay = defaults.timeOfDay;
    if (t.timeOfDay < 0 || t.timeOfDay >= 24)
    {
        // |days| < kMaxDays: the conversion is defined
        const double days = std::floor(t.timeOfDay / 24);
        t.timeOfDay -= days * 24;
        t.day = addDays(t.day, int64_t(days));
        if (t.timeOfDay >= 24 || t.timeOfDay < 0)
        {
            // Rounding at the boundary
            t.timeOfDay = 0;
        }
    }
}

void normaliseWeather(Weather& w)
{
    const Weather d;
    w.cloudiness    = clampFinite(w.cloudiness, 0, 1, d.cloudiness);
    w.rain          = clampFinite(w.rain, 0, 1, d.rain);
    w.fogDensity    = clampFinite(w.fogDensity, 0, kMaxFogDensity, d.fogDensity);
    w.snowCover     = clampFinite(w.snowCover, 0, 1, d.snowCover);
    w.waveAmplitude = clampFinite(w.waveAmplitude, 0, kMaxWaveAmplitude, d.waveAmplitude);
    w.waveSteepness = clampFinite(w.waveSteepness, 0, 1, d.waveSteepness);
    for (int i = 0; i < 2; ++i)
    {
        w.wind(i)        = clampFinite(w.wind(i), -kMaxWind, kMaxWind, d.wind(i));
        w.cloudOffset(i) = wrapPeriod(finiteOr(w.cloudOffset(i), d.cloudOffset(i)), kCloudTile);
    }
    w.validAt = finiteOr(w.validAt, d.validAt);
}

bool hasWorldTime(const World& world)
{
    return world.registry().isGlobal(WorldTime::id) && world.alive(world.worldStateEntity());
}

bool hasWeather(const World& world)
{
    return world.registry().isSpaceState(Weather::id);
}

bool setWorldTime(World& world, const WorldTimeChange& c, std::string& error, WorldTime* previous)
{
    if (!hasWorldTime(world))
    {
        error = "this world has no WorldTime (EnvironmentSetup::enabled)";
        return false;
    }
    if (c.hours && !checkRange("hours", *c.hours, 0, 24, true, error)) return false;
    if (c.rate && !checkRange("rate", *c.rate, -kMaxTimeRate, kMaxTimeRate, false, error)) return false;
    if (const std::string r = refusal(world, "WorldTime"); !r.empty())
    {
        error = r;
        return false;
    }
    WorldTime& t = world.global<WorldTime>();
    if (previous) *previous = t;
    normaliseWorldTime(t);
    // Continue from the time the extract shows: fold in the progress since validAt (zero with worldTimeSystem, which
    // writes validAt every step; in a world without it the stored time stays and only the extract extrapolates, so
    // a write of the rate alone would jump back to the stored time)
    if (!c.hours && t.rate != 0)
    {
        t.timeOfDay += t.rate * (world.time() - t.validAt) / 3600.0;
        normaliseWorldTime(t);
    }
    if (c.hours) t.timeOfDay = *c.hours;
    if (c.day) t.day = *c.day;
    if (c.rate) t.rate = *c.rate;
    t.validAt = world.time();
    SAIGA_LOG("api", "ok set_world_time time_of_day=%.4f day=%lld rate=%.4f", t.timeOfDay, (long long)t.day, t.rate);
    return true;
}

bool setWeather(World& world, SpaceId space, const WeatherChange& c, std::string& error, Weather* previous)
{
    if (!hasWeather(world))
    {
        error = "this world has no Weather (EnvironmentSetup::enabled)";
        return false;
    }
    if (space >= world.spaceCount() || !world.alive(world.spaceStateEntity(space)))
    {
        error =
            "space: no space " + std::to_string(space) + " (the world has " + std::to_string(world.spaceCount()) + ")";
        return false;
    }
    if ((c.cloudiness && !checkRange("cloudiness", *c.cloudiness, 0, 1, false, error)) ||
        (c.rain && !checkRange("rain", *c.rain, 0, 1, false, error)) ||
        (c.fogDensity && !checkRange("fog_density", *c.fogDensity, 0, kMaxFogDensity, false, error)) ||
        (c.snowCover && !checkRange("snow_cover", *c.snowCover, 0, 1, false, error)) ||
        (c.waveAmplitude && !checkRange("wave_amplitude", *c.waveAmplitude, 0, kMaxWaveAmplitude, false, error)) ||
        (c.waveSteepness && !checkRange("wave_steepness", *c.waveSteepness, 0, 1, false, error)))
        return false;
    if (c.wind)
        for (int i = 0; i < 2; ++i)
            if (!checkRange(i == 0 ? "wind[0]" : "wind[1]", (*c.wind)(i), -kMaxWind, kMaxWind, false, error))
                return false;
    if (c.cloudOffset)
        for (int i = 0; i < 2; ++i)
            if (!std::isfinite((*c.cloudOffset)(i)))
            {
                error = std::string(i == 0 ? "cloud_offset[0]" : "cloud_offset[1]") + ": expected a finite number";
                return false;
            }
    if (const std::string r = refusal(world, "Weather"); !r.empty())
    {
        error = r;
        return false;
    }
    Weather& w = world.spaceState<Weather>(space);
    if (previous) *previous = w;
    normaliseWeather(w);
    // Continue from the clouds the extract shows (as setWorldTime: the drift since validAt with the old wind)
    if (!c.cloudOffset)
        w.cloudOffset += w.wind * ((world.time() - w.validAt) * world.environmentSetup().cloudWindScale);
    if (c.cloudiness) w.cloudiness = *c.cloudiness;
    if (c.rain) w.rain = *c.rain;
    if (c.fogDensity) w.fogDensity = *c.fogDensity;
    if (c.snowCover) w.snowCover = *c.snowCover;
    if (c.waveAmplitude) w.waveAmplitude = *c.waveAmplitude;
    if (c.waveSteepness) w.waveSteepness = *c.waveSteepness;
    if (c.wind) w.wind = *c.wind;
    if (c.cloudOffset) w.cloudOffset = *c.cloudOffset;
    normaliseWeather(w);
    w.validAt = world.time();
    SAIGA_LOG("api", "ok set_weather space=%u cloudiness=%.3f rain=%.3f fog_density=%.6f", unsigned(space),
              w.cloudiness, w.rain, w.fogDensity);
    return true;
}

}  // namespace Engine
}  // namespace Saiga
