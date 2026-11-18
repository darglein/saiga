/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

namespace Saiga
{
namespace Engine
{
class World;

/**
 * Phase 6, after transformSystem (docs/engine_design.md §9.5). Reads: the structural log (its own consumer: created
 * entities enter the grid, destroyed ones leave it), the world's list of changed transforms and their Transform.
 * Writes: the world's SpatialGrid (an entity's cell and stored position). Idle entities cost nothing.
 */
SAIGA_ENGINE_API void spatialGridSystem(World& world, double dt);

// Rebuilds the grid from every alive entity (engine-internal: after the registry was filled directly, e.g. load in S11;
// between steps)
SAIGA_ENGINE_API void rebuildSpatialGrid(World& world);

}  // namespace Engine
}  // namespace Saiga
