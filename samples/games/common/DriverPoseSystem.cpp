/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// driverPoseSystem (S16b). Reads: Driver, the vehicle's Transform (host) or RenderPose (client). Writes: the owned
// drivers' Transform.

#include "saiga/engine/components/RenderPose.h"

#include "games/common/Driving.h"

namespace Saiga
{
namespace Game
{
using namespace Engine;

void driverPoseSystem(World& w, double)
{
    w.ownedBy<Transform, const Driver>(w.localPeer())
        .each(
            [&](Entity e, Transform& t, const Driver& d)
            {
                // An AI vehicle drives itself
                if (d.vehicle == e || !d.vehicle.valid() || !w.alive(d.vehicle)) return;
                if (!w.isHost() && w.has<RenderPose>(d.vehicle))
                {
                    const RenderPose& p = w.read<RenderPose>(d.vehicle);
                    t.setPosition(p.position);
                    t.setRotation(p.rotation);
                    return;
                }
                const Transform& v = w.read<Transform>(d.vehicle);
                t.setPosition(v.position());
                t.setRotation(v.rotation());
            });
}

}  // namespace Game
}  // namespace Saiga
