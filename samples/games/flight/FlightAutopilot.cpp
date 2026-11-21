/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Autopilot of the flight simulator demo mode (circuit, height and speed hold).

#include "games/flight/FlightAutopilot.h"

namespace Saiga
{
namespace Game
{
AircraftControls FlightAutopilot::controls(const FlightModel& m) const
{
    const auto& s = m.state;
    Vec3 r(s.position(0) - center(0), 0, s.position(2) - center(2));
    double d = r.norm();
    Vec3 rn  = d > 1e-6 ? Vec3(r / d) : Vec3(1, 0, 0);
    // Tangent of the circle, corrected towards the radius
    Vec3 tangent = Vec3(-rn(2), 0, rn(0)) * (counterClockwise ? 1.0 : -1.0);
    Vec3 desired = (tangent - rn * clampd((d - radius) / 400.0, -1, 1)).normalized();

    Vec3 f(m.forward()(0), 0, m.forward()(2));
    double headingError = signedAngleXZ(f, desired);
    // A left turn (positive error) needs a left bank (negative bank angle)
    double bankTarget = clampd(-headingError * 1.5, -maxBank, maxBank);

    AircraftControls c;
    c.roll = clampd((bankTarget - m.bankAngle()) * 2.0 - s.bodyRates(2) * -0.3, -1, 1);

    // Height above the highest ground near the plane
    double ground   = m.groundHeight();
    double ahead    = m.ground ? m.ground(s.position(0) + f(0) * 400, s.position(2) + f(2) * 400) : ground;
    double targetY  = std::max(ground, ahead) + height;
    double vsTarget = clampd((targetY - s.position(1)) * 0.15, -8, 12);
    c.pitch         = clampd((vsTarget - s.velocity(1)) * 0.06 + std::abs(m.bankAngle()) * 0.25, -1, 1);

    // Same sign as the model's yaw stability (FlightModel::step: target -= yawRate * yaw + yawStability * beta), so
    // the rudder helps to remove the sideslip
    c.yaw = clampd(m.sideSlip() * 3.0, -1, 1);
    return c;
}

double FlightAutopilot::throttle(const FlightModel& m) const
{
    return clampd(m.trimThrottle(speed) + (speed - m.airspeed()) * 0.08, 0, 1);
}

}  // namespace Game
}  // namespace Saiga
