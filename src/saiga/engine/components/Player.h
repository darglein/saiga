/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"

namespace Saiga
{
namespace Engine
{
// A player character (docs/engine_design.md §9.3): used by ticking (§8.4), later streaming, interest and ownership.
// peer is a session slot (0 before S16); characterId (!= 0) names the player across sessions (piece owners, game
// references store it, never a peer index).
struct Player
{
    static constexpr const char* name = "Player";
    static constexpr ComponentId id   = ComponentId(EngineComponent::Player);
    static constexpr uint32_t version = 1;

    uint8_t peer         = 0;
    uint64_t characterId = 0;
    double moveSpeed     = 5;

    template <class V>
    void visit(V& v)
    {
        v("peer", peer);
        v("characterId", characterId);
        v("moveSpeed", moveSpeed);
    }
};

}  // namespace Engine
}  // namespace Saiga
