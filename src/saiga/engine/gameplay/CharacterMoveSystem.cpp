/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// characterMoveSystem (phase 4, both; §8.2, §9.3): kinematic until S29. Reads: CharacterIntent. Writes: Transform
// (position += desiredVelocity * dt; y = the ground height, 0 before S20). Idle characters are not written.

#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/gameplay/Systems.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
void characterMoveSystem(World& w, double dt)
{
    w.view<Transform, const CharacterIntent>().each(
        [dt](Entity, Transform& t, const CharacterIntent& intent)
        {
            if (intent.desiredVelocity == Vec3::Zero()) return;
            Vec3 p = t.position() + intent.desiredVelocity * dt;
            p.y()  = 0;  // flat ground before S20
            t.setPosition(p);
        });
}

}  // namespace Engine
}  // namespace Saiga
