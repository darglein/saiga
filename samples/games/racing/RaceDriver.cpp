/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// AI driver of the racing demo mode.

#include "games/racing/RaceDriver.h"

namespace Saiga
{
namespace Game
{
VehicleControls RaceDriver::controls(const Vehicle& car, const Track& track)
{
    VehicleControls c;
    const auto& p = car.state.position;
    progress      = track.closest(p, progress);
    double speed  = car.forwardSpeed();

    Vec3 target = track.position(progress + lookAhead + std::max(speed, 0.0) * 0.5);
    Vec3 to(target(0) - p(0), 0, target(2) - p(2));
    Vec3 f(car.forward()(0), 0, car.forward()(2));
    // Positive angle: target to the left, so steer left (negative)
    double angle = signedAngleXZ(f, to);
    c.steer      = clampd(-angle * 2.5, -1, 1);

    // Speed limit: every corner ahead within the braking distance must be reachable at its corner speed
    // (v^2 <= v_corner^2 + 2 * braking * d)
    double limit = maxSpeed;
    double reach = speed * speed / (2 * braking) + 10;
    for (double d = 0; d < reach; d += 2)
    {
        double k = track.curvature(progress + d);
        if (k > 1e-6)
        {
            double vCorner = std::sqrt(lateralAcceleration / k);
            limit          = std::min(limit, std::sqrt(vCorner * vCorner + 2 * braking * d));
        }
    }
    c.throttle = speed < limit ? 1.0 : (speed > limit + 2 ? -0.6 : 0.2);
    return c;
}

}  // namespace Game
}  // namespace Saiga
