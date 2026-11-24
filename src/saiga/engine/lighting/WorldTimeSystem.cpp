/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// worldTimeSystem: writes WorldTime (global) and Weather (space state of every space); reads
// EnvironmentSetup::cloudWindScale.

#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/lighting/Systems.h"
#include "saiga/engine/world/SystemList.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
namespace
{
bool same(const WorldTime& a, const WorldTime& b)
{
    return a.timeOfDay == b.timeOfDay && a.day == b.day && a.rate == b.rate && a.validAt == b.validAt;
}
bool same(const Weather& a, const Weather& b)
{
    return a.cloudiness == b.cloudiness && a.rain == b.rain && a.fogDensity == b.fogDensity &&
           a.snowCover == b.snowCover && a.waveAmplitude == b.waveAmplitude && a.waveSteepness == b.waveSteepness &&
           a.wind == b.wind && a.cloudOffset == b.cloudOffset && a.validAt == b.validAt;
}
}  // namespace

void worldTimeSystem(World& world, double dt)
{
    // The values written in this step are valid at its end (World::time() is the step's start during the step). A
    // component is written (and so replicated and changed for the save diff) only when its value changes: with rate 0
    // and no wind nothing moves, and the old validAt still gives the same extrapolation
    const double validAt = world.time() + dt;
    if (hasWorldTime(world))
    {
        const WorldTime& stored = world.readGlobal<WorldTime>();
        WorldTime t             = stored;
        normaliseWorldTime(t);
        if (t.rate != 0)
        {
            t.timeOfDay += t.rate * dt / 3600.0;
            normaliseWorldTime(t);
            t.validAt = validAt;
        }
        if (!same(t, stored)) world.global<WorldTime>() = t;
    }
    if (hasWeather(world))
    {
        const double scale = world.environmentSetup().cloudWindScale;
        for (size_t space = 0; space < world.spaceCount(); ++space)
        {
            const Entity e = world.spaceStateEntity(SpaceId(space));
            if (!world.alive(e)) continue;
            const Weather& stored = world.read<Weather>(e);
            Weather w             = stored;
            normaliseWeather(w);
            const Vec2 drift = w.wind * (dt * scale);
            if (drift != Vec2::Zero())
            {
                w.cloudOffset += drift;
                normaliseWeather(w);
                w.validAt = validAt;
            }
            if (!same(w, stored)) world.get<Weather>(e) = w;
        }
    }
}

void addEnvironmentSystems(SystemList& systems)
{
    systems.add(Phase::Gameplay, "world_time", worldTimeSystem, RunOn::Host);
}

}  // namespace Engine
}  // namespace Saiga
