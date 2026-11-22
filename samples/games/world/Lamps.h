/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/world/World.h"

#include <functional>

namespace Saiga
{
namespace Game
{
/**
 * The lamps of game_world (S36b, B7 of docs/stages/S36_lighting.plan.md; game policy, R-MECH-1): lamp posts with a
 * warm point light (1200 lm, range 18 m; every 8th casts shadows: the atlas's tile budget decides which get tiles),
 * street lamps with a spot light pointing down (25 / 35 degrees) and flickering torches, on a grid along the terrain
 * around a point (the heights from the terrain; none below the sea). Entities with a RenderInstance (the post, its
 * glowing head) and a LightSource; lampPolicySystem switches them on while the sun is below 4 degrees (they replicate
 * with their entities).
 */
// The meshes of the lamps (WorldSetup::renderAssets)
void lampRenderAssets(Engine::World& world);
// Spawns count lamps on a grid of spacing m centred at centre (x, z), or (ringRadius > 0) evenly on a circle of that
// radius around it; the ground from ground(x, z), none below seaLevel; returns the lamps spawned
int spawnLamps(Engine::World& world, int count, double centreX, double centreZ, double spacing, double ringRadius,
               const std::function<double(double, double)>& ground, double seaLevel);
// Phase Gameplay (host): every LightSource on while the sun is below 4 degrees (the world's time and celestial model)
void lampPolicySystem(Engine::World& world, double dt);

}  // namespace Game
}  // namespace Saiga
