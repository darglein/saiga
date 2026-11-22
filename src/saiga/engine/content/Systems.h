/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The content systems (docs/engine_design.md §8.2, §12.1; S23a).

#pragma once

#include "saiga/export.h"

namespace Saiga
{
namespace Engine
{
class World;

// Phase 7 (Streaming), RunOn::Host, the first system of the phase; registered by the World only when some plan has
// sites or a generator in use is not legacy (a load may bring plans with sites). Reads Player, Transform and the plans; stamps the prefab entities of every chunk within the space's
// fullRadius of a Player (spawns through commands, SpaceLink from the plan) and writes the chunk records.
SAIGA_ENGINE_API void chunkStampSystem(World& world, double dt);

}  // namespace Engine
}  // namespace Saiga
