/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The space template of the world prototype (S20, docs/engine_design.md §12.2, A6): the 32 km space of game_world, the
// flight and the racing demo at 1 m (2 m for the benchmark comparison run) with the prototype's sea level and height
// hook. It lives in the games (D2: the hook is the prototype's game side in vulkan/world/PrototypeTerrain).

#pragma once

#include "saiga/engine/world/SpaceTemplate.h"
#include "saiga/vulkan/world/PrototypeTerrain.h"

namespace Saiga
{
namespace Game
{
inline Engine::SpaceTemplate prototypeSpaceTemplate(int baseSpacing = 1)
{
    Engine::SpaceTemplate t;
    t.name            = "prototype";
    t.kind            = Engine::SpaceKind::OpenWorld;
    t.baseSpacing     = baseSpacing;
    t.seaLevel        = Vulkan::kPrototypeSeaLevel;
    t.sizeChunks      = Vulkan::kPrototypeSpaceChunks;
    t.prototypeHeight = Vulkan::prototypeHeight;
    return t;
}

}  // namespace Game
}  // namespace Saiga
