/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// streamingSystem (phase 7, both; S24): the residency update. Results applied in a step whose later phase aborts stay
// applied (derived data; the next step continues).

#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/streaming/Systems.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
void streamingSystem(World& world, double)
{
    world.residency().update();
}

}  // namespace Engine
}  // namespace Saiga
