/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// characterMoveSystem (phase 4, both; §8.2, §9.3): kinematic until S29, over the entities this peer owns (R-NET-2).
// Reads: CharacterIntent. Writes: Transform (position += desiredVelocity * dt; y = World::terrainHeight, S20).
// Idle characters (zero intent) in a space with terrain are written only when their y differs from terrainHeight at
// their x, z (a spawned or teleported character is snapped in the next step, a resting one is never dirtied); in a
// space without terrain they are not written. A character legitimately above the surface does not exist before
// collision (S28 replaces the mover).

#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/gameplay/Systems.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
void characterMoveSystem(World& w, double dt)
{
    const bool streaming = w.streaming();
    uint64_t holds       = 0;
    w.ownedBy<Transform, const CharacterIntent>(w.localPeer())
        .each(
            [&w, dt, streaming, &holds](Entity, Transform& t, const CharacterIntent& intent)
            {
                const SpaceId space = t.space();
                const Vec3& from    = t.position();
                // S24 (G11): in a streaming world nothing is written while the current or the destination chunk is not
                // requested and resident on this peer (chunkReady clamps to the space's chunk range)
                if (streaming)
                {
                    const Vec3 to = from + intent.desiredVelocity * dt;
                    if (!w.chunkReady(space, from.x(), from.z()) || !w.chunkReady(space, to.x(), to.z()))
                    {
                        holds++;
                        return;
                    }
                }
                if (intent.desiredVelocity == Vec3::Zero())
                {
                    if (!w.spaceHasTerrain(space)) return;
                    const Vec3& p  = t.position();
                    const double y = w.terrainHeight(space, p.x(), p.z());
                    if (p.y() != y) t.setPosition(Vec3(p.x(), y, p.z()));
                    return;
                }
                Vec3 p = t.position() + intent.desiredVelocity * dt;
                p.y()  = w.terrainHeight(space, p.x(), p.z());
                t.setPosition(p);
            });
    if (holds > 0) w.residency().statsForUpdate().moverHolds += holds;
}

}  // namespace Engine
}  // namespace Saiga
