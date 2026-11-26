/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// simulationSystem (phase 7, both; S24, G8): the Simulated evaluation right after the residency update, so phases 2-5
// of the next step see flags computed from the current positions and the current ready set.

#include "saiga/engine/streaming/SimulationSet.h"
#include "saiga/engine/streaming/Systems.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
void simulationSystem(World& world, double)
{
    world.simulationSet().update();
}

}  // namespace Engine
}  // namespace Saiga
