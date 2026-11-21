/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Entity.h"
#include "saiga/export.h"

// The engine's gameplay systems (docs/engine_design.md §8.2, §8.4, §9.2, §9.3), registered by registerEngineSystems.
// One file per system; reads and writes are listed at each definition.

namespace Saiga
{
namespace Engine
{
class World;
struct Behaviour;

// Phase 2 (host): DamageEvent, InteractEvent and ContactEvent of the previous step to the game objects' handlers
SAIGA_ENGINE_API void behaviourEventSystem(World& world, double dt);
// Phase 3 (host): GameObject::update for the objects whose tick is due
SAIGA_ENGINE_API void behaviourSystem(World& world, double dt);
// Phase 4 (both): PlayerInput -> CharacterIntent of the owned players
SAIGA_ENGINE_API void playerControlSystem(World& world, double dt);
// Phase 4 (both): kinematic movement by CharacterIntent::desiredVelocity on flat ground (y = 0 before S20)
SAIGA_ENGINE_API void characterMoveSystem(World& world, double dt);
// Phase 6 (all), after the grid: tick intervals from the grid-cell rings around the players
SAIGA_ENGINE_API void tickSystem(World& world, double dt);
// Phase 4 (host, optional: games register it, R-MECH-1): an InteractEvent of the previous step whose target has a
// SpaceLink, whose actor is an alive root in the target's space, requests the transition (S21)
SAIGA_ENGINE_API void spaceLinkSystem(World& world, double dt);

// The tick interval of one object from the current player cells; spawned: also the staggered first tick (tickOffset)
SAIGA_ENGINE_API void assignTickInterval(World& world, Entity e, Behaviour& behaviour, bool spawned);

}  // namespace Engine
}  // namespace Saiga
