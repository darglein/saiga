/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/tools/ScenarioList.h"
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
 * around a point (the heights from the terrain; none below the sea). Entities with a RenderInstance: the post and
 * its head (the torch and its flame); the head (the flame, the street lamp itself) carries the LightSource.
 * lampPolicySystem switches them on at dusk (the sun below 4 degrees) and off at dawn (they replicate with their
 * entities) and gives a head or flame its glowing mesh only while its light is on (round 1, finding 18: the heads
 * glowed by day). It drives only the lamps spawnLamps spawned (LampPolicy) and writes their light only on a day/night
 * transition: a light switched by hand (the Lighting tab, set_component) keeps its state until the next dusk or dawn,
 * a light added by hand is never switched (round 4, minor 1: the policy rewrote every LightSource's `on` each step).
 */
// The policy's state of a lamp spawnLamps spawned (on the entity with the LightSource; host only, saved)
struct LampPolicy
{
    static constexpr const char* name       = "LampPolicy";
    static constexpr Engine::ComponentId id = Engine::kFirstGameComponent;
    static constexpr uint32_t version       = 1;
    // The day/night state last applied: 1 on, 0 off, -1 none yet (the first step applies the current one)
    int8_t applied = -1;
    template <class V>
    void visit(V& v)
    {
        v("applied", applied);
    }
};
// Registers LampPolicy (WorldSetup::components)
void lampComponents(Engine::Registry& registry);
// The meshes of the lamps (WorldSetup::renderAssets)
void lampRenderAssets(Engine::World& world);
// Spawns count lamps on a grid of spacing m centred at centre (x, z), or (ringRadius > 0) evenly on a circle of that
// radius around it; the ground from ground(x, z), none below seaLevel; returns the lamps spawned
int spawnLamps(Engine::World& world, int count, double centreX, double centreZ, double spacing, double ringRadius,
               const std::function<double(double, double)>& ground, double seaLevel);
// Phase Gameplay (host): on a day/night transition (the sun below 4 degrees: night; the world's time and celestial
// model) the LightSource of every lamp with a LampPolicy on or off; a lamp head or torch flame glows while its light is
// on
void lampPolicySystem(Engine::World& world, double dt);
// game_world's scenarios (--scenarios): lamp_hand_switch_holds
void lampScenarios(Engine::ScenarioList& list);

}  // namespace Game
}  // namespace Saiga
