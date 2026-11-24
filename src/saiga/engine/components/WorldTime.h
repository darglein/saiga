/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// The time of day of the world (docs/engine_design.md §7.8, §10.7; S36): a global (world-state component) when
// WorldSetup::environment.enabled, else a plain component no entity has. Advanced by the optional worldTimeSystem
// (addEnvironmentSystems, R-MECH-1); written by setWorldTime (tools, the Lighting tab). Replicate::Host, saved.
struct WorldTime
{
    static constexpr const char* name = "WorldTime";
    static constexpr ComponentId id   = ComponentId(EngineComponent::WorldTime);
    static constexpr uint32_t version = 1;

    double timeOfDay = 12;  // hours, [0, 24)
    int64_t day      = 0;
    // Game seconds per simulated second, a time multiplier (0: frozen; 60: a game hour per minute; 86400 / s: a day in
    // s seconds), [-3600, 3600]; timeOfDay advances by rate * dt / 3600 hours per step
    double rate = 0;
    // World::time() at which timeOfDay is valid (the step that wrote it): the reference of the render time's
    // interpolation (host) and extrapolation (client)
    double validAt = 0;

    template <class V>
    void visit(V& v)
    {
        v("timeOfDay", timeOfDay);
        v("day", day);
        v("rate", rate);
        v("validAt", validAt);
    }
};

}  // namespace Engine
}  // namespace Saiga
