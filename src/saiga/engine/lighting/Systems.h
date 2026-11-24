/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

// The environment's systems (docs/engine_design.md §8.2, §10.7; S36). Optional: a game registers them with
// addEnvironmentSystems (R-MECH-1: how time passes is the game's choice).

namespace Saiga
{
namespace Engine
{
class World;
class SystemList;

// Phase 4 (host): advances WorldTime by rate * dt (wrapped into [0, 24) with the day count) and every space's
// Weather::cloudOffset by wind * dt * EnvironmentSetup::cloudWindScale (wrapped into [0, kCloudTile)); normalises both
// every step (regardless of rate) and sets their validAt. Nothing without EnvironmentSetup::enabled
SAIGA_ENGINE_API void worldTimeSystem(World& world, double dt);

// Registers worldTimeSystem (phase Gameplay, RunOn::Host)
SAIGA_ENGINE_API void addEnvironmentSystems(SystemList& systems);

}  // namespace Engine
}  // namespace Saiga
