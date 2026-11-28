/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The unloading systems (docs/engine_design.md §8.2, §9.6, §12.4; S25), registered by the World constructor with
// WorldConfig::streaming (G18), in phase 7: streaming, simulation (S24), space_load, chunk_stamp, chunk_entities,
// space_unload. Host only (clients get entities through replication: Despawn and a new spawn).

#pragma once

#include "saiga/export.h"

namespace Saiga
{
namespace Engine
{
class World;

// Phase 7, host, before chunk_stamp: recreates the state entities of unloaded spaces that got a streaming centre (a
// command at the flush). Reads: the residency's centres, the chunk store. Writes: the space-state entity, the store
SAIGA_ENGINE_API void spaceLoadSystem(World& world, double dt);
// Phase 7, host, after chunk_stamp: the reloads of chunks that became ready and the records of chunks that became
// resident, then the one unload pass (every unloadable root whose chunk is not kept; commands at the flush). Reads:
// Transform, Parent, Player, PendingTransition, NetOwner, Behaviour, the spatial grid, the residency. Writes: the
// entities of stored groups, the chunk store, the chunk records
SAIGA_ENGINE_API void chunkEntitySystem(World& world, double dt);
// Phase 7, host, after chunk_entities: a loaded space without centres and without roots unloads (its state entity into
// the space record; a command at the flush)
SAIGA_ENGINE_API void spaceUnloadSystem(World& world, double dt);

}  // namespace Engine
}  // namespace Saiga
