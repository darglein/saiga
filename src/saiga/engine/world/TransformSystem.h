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
class SystemList;

/**
 * Phase 6 (docs/engine_design.md §7.4). Reads: Parent, LocalTransform, Transform of the parents. Writes: Transform of
 * the children (through the setters; a teleported parent teleports its children) and the world's list of changed
 * transforms (changed in this or the previous step, sorted by index), which the grid (S09) and extract (S19) read.
 * Children are updated in depth order (the list is rebuilt when Parent components or entities change).
 */
SAIGA_ENGINE_API void transformSystem(World& world, double dt);

// The engine's systems (docs/engine_design.md §8.2), registered before the game's
SAIGA_ENGINE_API void registerEngineSystems(SystemList& systems);

}  // namespace Engine
}  // namespace Saiga
