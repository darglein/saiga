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
void worldTimeSystem(World& world, double dt)
{
    // The values written in this step are valid at its end (World::time() is the step's start during the step)
    const double validAt = world.time() + dt;
    if (hasWorldTime(world))
    {
        WorldTime& t = world.global<WorldTime>();
        normaliseWorldTime(t);
        t.timeOfDay += t.rate * dt / 3600.0;
        normaliseWorldTime(t);
        t.validAt = validAt;
    }
    if (hasWeather(world))
    {
        const double scale = world.environmentSetup().cloudWindScale;
        for (size_t space = 0; space < world.spaceCount(); ++space)
        {
            const Entity e = world.spaceStateEntity(SpaceId(space));
            if (!world.alive(e)) continue;
            Weather& w = world.get<Weather>(e);
            normaliseWeather(w);
            w.cloudOffset += w.wind * (dt * scale);
            normaliseWeather(w);
            w.validAt = validAt;
        }
    }
}

void addEnvironmentSystems(SystemList& systems)
{
    systems.add(Phase::Gameplay, "world_time", worldTimeSystem, RunOn::Host);
}

}  // namespace Engine
}  // namespace Saiga
