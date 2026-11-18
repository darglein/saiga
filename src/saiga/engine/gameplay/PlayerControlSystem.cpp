/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// playerControlSystem (phase 4, both; §9.3). Reads: Player (moveSpeed), PlayerInput of the owned players. Writes:
// CharacterIntent::desiredVelocity = move * moveSpeed (move longer than 1 is shortened to 1).

#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/gameplay/Systems.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
void playerControlSystem(World& w, double)
{
    w.ownedBy<const Player, const PlayerInput, CharacterIntent>(w.localPeer())
        .each(
            [](Entity, const Player& p, const PlayerInput& in, CharacterIntent& intent)
            {
                Vec3 move           = in.move;
                const double length = move.norm();
                if (!(length <= 1)) move = length > 1 && std::isfinite(length) ? Vec3(move / length) : Vec3::Zero();
                intent.desiredVelocity = move * p.moveSpeed;
            });
}

}  // namespace Engine
}  // namespace Saiga
