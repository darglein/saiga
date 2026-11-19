/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"
#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"

namespace Saiga
{
namespace Engine
{
// The latest input of a player (docs/engine_design.md §9.3; latest wins, written by the owning peer): move direction
// in world space (length <= 1), look rotation and held buttons (game-defined bits). One-shot actions are PlayerActions.
struct PlayerInput
{
    static constexpr const char* name = "PlayerInput";
    static constexpr ComponentId id   = ComponentId(EngineComponent::PlayerInput);
    static constexpr uint32_t version = 1;

    Vec3 move            = Vec3::Zero();
    Quat look            = Quat::Identity();
    uint32_t heldButtons = 0;

    template <class V>
    void visit(V& v)
    {
        v("move", move);
        v("look", look);
        v("heldButtons", heldButtons);
    }
};

}  // namespace Engine
}  // namespace Saiga
