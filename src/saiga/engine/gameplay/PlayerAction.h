/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/Types.h"
#include "saiga/engine/ecs/Entity.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// A one-shot action of a player (docs/engine_design.md §9.3): game-defined action code and aim. Before S15 queued
// locally with world.queueAction; from S15 a reliable message. Delivered as ActionEvent (readable in the next step) in
// (peer, sequence) order, exactly once.
struct PlayerAction
{
    Entity player;
    uint32_t action = 0;
    Vec3 aim        = Vec3::Zero();
};

}  // namespace Engine
}  // namespace Saiga
