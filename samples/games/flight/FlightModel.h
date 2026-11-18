/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/common/GameMath.h"

namespace Saiga
{
namespace Game
{
// A small propeller airplane (SI units)
struct AircraftParameters
{
    double mass       = 1200;
    double wingArea   = 16;
    double maxThrust  = 6500;
    double airDensity = 1.225;
    // Lift coefficient: cl0 + clAlpha * alpha up to the stall angle, then it drops
    double cl0        = 0.35;
    double clAlpha    = 5.0;
    double stallAngle = 15.0 * kPi / 180.0;
    // Drag coefficient: cd0 + inducedDrag * cl^2
    double cd0         = 0.03;
    double inducedDrag = 0.05;
    // Side force per m/s of side slip velocity (N)
    double sideForce = 600;
    // Body rates (rad/s) at full control deflection and the reference airspeed
    double pitchRate = 1.0, rollRate = 2.2, yawRate = 0.5;
    double referenceSpeed = 55;
    // Aerodynamic stability: rates (rad/s per rad) that turn the nose into the air flow
    double pitchStability = 3.0, yawStability = 2.0;
    // How fast the body rates follow their targets (1/s)
    double rateResponse = 6.0;
    // Height of the fuselage above the ground when standing on the gear
    double gearHeight = 1.5;
    // Touchdown limits: faster sinking or more bank / pitch is a crash
    double crashSinkRate = 6.0, crashBank = 30.0 * kPi / 180.0, crashPitch = 25.0 * kPi / 180.0;
};

// Pilot inputs, all in [-1, 1] except the throttle [0, 1]. Positive: pitch nose up, roll right, yaw right.
struct AircraftControls
{
    double throttle = 0.5, pitch = 0, roll = 0, yaw = 0;
};

struct AircraftState
{
    Vec3 position    = Vec3(0, 0, 0);
    Vec3 velocity    = Vec3(0, 0, 0);
    Quat orientation = Quat::Identity();
    // Body rates: x pitch (nose up), y yaw (nose left), z roll (right wing up)
    Vec3 bodyRates = Vec3(0, 0, 0);
    bool onGround  = false;
    bool crashed   = false;
};

/**
 * Flight model of the flight simulator tech demo (docs/engine_design.md §18.4), double precision, fixed step.
 * Forces: thrust along the nose, lift perpendicular to the velocity (coefficient linear in the angle of attack,
 * dropping after the stall angle), drag (parasitic + induced), side force against side slip, gravity. Rotation: the
 * body rates follow the control inputs (scaled with the airspeed) plus a stability term that turns the nose into the
 * air flow; no full inertia tensor. Ground: the gear keeps the plane at gearHeight above the ground; touching down too
 * hard, too steep or banked crashes it.
 */
class FlightModel
{
   public:
    AircraftParameters params;
    AircraftState state;
    // Water and terrain: the plane crashes into max(terrain, water)
    GroundHeight ground;

    // Airborne at position, heading (see yawRotation) and airspeed, wings level
    void reset(const Vec3& position, double heading, double speed);
    void step(const AircraftControls& controls, double dt);

    Vec3 forward() const { return forwardOf(state.orientation); }
    Vec3 up() const { return upOf(state.orientation); }
    Vec3 right() const { return rightOf(state.orientation); }

    double airspeed() const { return state.velocity.norm(); }
    // Angle between the nose and the air flow in the symmetry plane (positive: flow from below)
    double angleOfAttack() const;
    // Angle between the nose and the air flow sideways (positive: flow from the right)
    double sideSlip() const;
    // Right wing down: positive
    double bankAngle() const;
    double pitchAngle() const;
    double liftCoefficient(double alpha) const;
    bool stalled() const { return std::abs(angleOfAttack()) > params.stallAngle; }
    double groundHeight() const;
    double altitudeAboveGround() const { return state.position(1) - groundHeight(); }

    // Speed of level flight with alpha = 0 (lift = weight) and the throttle that holds it
    double trimSpeed() const;
    double trimThrottle(double speed) const;
};

}  // namespace Game
}  // namespace Saiga
