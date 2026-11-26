/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The streaming systems (docs/engine_design.md §8.2, §12.3; S24), registered by the World constructor with
// WorldConfig::streaming, in phase 7 before chunk_stamp: streaming, then simulation.

#pragma once

#include "saiga/export.h"

namespace Saiga
{
namespace Engine
{
class World;

// Phase 7, both roles: World::residency().update() (centres, requests, unloads, the pool's completions applied, the
// stall counters; the host's islands and IslandLimitEvent). Reads: Player, StreamingAnchor, PendingTransition,
// Transform, Parent, the chunk records, the terrain sequences. Writes: the residency (derived), the pinned terrain
// tiles, IslandState (a command at the flush)
SAIGA_ENGINE_API void streamingSystem(World& world, double dt);
// Phase 7, both roles, after streaming: the Simulated tag (G8; a client: the held roots only). Reads: Transform,
// Parent, Player, StreamingAnchor, PendingTransition, Behaviour, the spatial grid, the residency. Writes: the entity
// flags
SAIGA_ENGINE_API void simulationSystem(World& world, double dt);

}  // namespace Engine
}  // namespace Saiga
