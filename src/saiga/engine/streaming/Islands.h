/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Islands (docs/engine_design.md §12.3; S24, G14): per space the connected components of the player roots (G23, in
// entity-index order; a pair is connected when its horizontal distance is at most 2 * simulationRadius of the space),
// summed over the spaces. Pure functions over the World's players (the host's streaming system, requestArrival's
// hard limit, tests).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/Types.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/export.h"

namespace Saiga
{
namespace Engine
{
class World;

struct SAIGA_ENGINE_API Islands
{
    // The island count of the current player roots at their current positions
    static int count(const World& world);
    // The count once every accepted arrival completed: roots with a pending arrival at their target (hardIslandLimit)
    static int countAccepted(const World& world);
    // countAccepted with the root `moved` at (space, position) (a player root not yet counted is added; requestArrival)
    static int countWith(const World& world, Entity moved, SpaceId space, const Vec3& position);
};

}  // namespace Engine
}  // namespace Saiga
