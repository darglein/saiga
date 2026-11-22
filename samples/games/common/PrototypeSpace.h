/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The space template of the world prototype (S20, docs/engine_design.md §12.2, A6): the 32 km space of game_world, the
// flight and the racing demo at 1 m (2 m for the benchmark comparison run) with the prototype's sea level and height
// hook. It lives in the games (D2: the hook is the prototype's game side in vulkan/world/PrototypeTerrain). S23a: the
// hook is the HookGenerator "prototype" (float noise, not integer only: the prototype fixture until S27), which a game
// registers with registerPrototypeGenerator.

#pragma once

#include "saiga/engine/content/GeneratorList.h"
#include "saiga/engine/terrain/HookGenerator.h"
#include "saiga/engine/world/SpaceTemplate.h"
#include "saiga/vulkan/world/PrototypeTerrain.h"

namespace Saiga
{
namespace Game
{
// The prototype's generator ID in the games' generator lists
constexpr Engine::GeneratorId kPrototypeGenerator = 1;

// The prototype hook as a generator (S20's grass thresholds; its masks are used for the content hashes only, the
// rendering keeps TerrainSystem's grass rule until S23b)
inline std::shared_ptr<const Engine::Generator> prototypeGenerator()
{
    return std::make_shared<Engine::HookGenerator>("prototype", Vulkan::prototypeHeight,
                                                   Engine::HookGrassRule{0.82, 0.4}, false);
}
// WorldSetup::generators of a game on the prototype space
inline void registerPrototypeGenerator(Engine::GeneratorList& list)
{
    list.add(kPrototypeGenerator, prototypeGenerator());
}

inline Engine::SpaceTemplate prototypeSpaceTemplate(int baseSpacing = 1)
{
    Engine::SpaceTemplate t;
    t.name        = "prototype";
    t.kind        = Engine::SpaceKind::OpenWorld;
    t.baseSpacing = baseSpacing;
    t.seaLevel    = Vulkan::kPrototypeSeaLevel;
    t.sizeChunks  = Vulkan::kPrototypeSpaceChunks;
    t.generator   = kPrototypeGenerator;
    return t;
}

}  // namespace Game
}  // namespace Saiga
