/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/common/GameMath.h"

#include <array>

namespace Saiga
{
namespace Game
{
struct VehicleParameters
{
    double mass = 1200;
    // Half extents of the chassis box (inertia), forward is -z
    Vec3 halfExtents = Vec3(0.9, 0.45, 2.1);
    // Wheel mounts relative to the center of mass: front left, front right, rear left, rear right
    std::array<Vec3, 4> wheelMounts = {Vec3(-0.82, -0.2, -1.35), Vec3(0.82, -0.2, -1.35), Vec3(-0.82, -0.2, 1.3),
                                       Vec3(0.82, -0.2, 1.3)};
    double wheelRadius              = 0.36;
    // Suspension: rest length below the mount, spring (N/m) and damper (Ns/m)
    double restLength = 0.35, stiffness = 38000, damping = 4200;
    // Drive (rear wheels), brakes, steering
    double engineForce = 7500, brakeForce = 11000, maxSpeed = 60, maxSteer = 0.55;
    // Tire friction coefficient and the slip speed (m/s) at which the full friction is reached
    double friction = 1.2, slipSpeed = 0.6;
    // The tire forces act at the roll centre, this far below the center of mass (instead of at the contact point,
    // 0.8 m below): with a half track of 0.82 m the car would roll over at about 1 g, below the tire friction)
    double rollCenterDepth = 0.3;
    // Friction of the chassis on the ground (on its side or roof)
    double chassisFriction = 0.8;
    // The body below and above the center of mass (the drawn car: floor 0.35 m below, roof 0.84 m above); with the
    // half width and length of halfExtents they keep a tipped car on top of the ground
    double bodyBottom = 0.35, bodyTop = 0.84;
    double rollingResistance = 25, airDrag = 0.45;
    // Sub steps per step (the suspension is stiff)
    int subSteps = 4;
};

// Driver inputs: throttle in [-1, 1] (negative: brake, then reverse), steer in [-1, 1] (positive: right)
struct VehicleControls
{
    double throttle = 0, steer = 0;
};

struct VehicleState
{
    Vec3 position    = Vec3(0, 0, 0);
    Vec3 velocity    = Vec3(0, 0, 0);
    Quat orientation = Quat::Identity();
    // World frame
    Vec3 angularVelocity              = Vec3(0, 0, 0);
    std::array<double, 4> compression = {};
    std::array<bool, 4> contact       = {};
    // Visual: wheel spin angle (rad) and the current steering angle
    std::array<double, 4> spin = {};
    double steerAngle          = 0;
};

/**
 * Vehicle model of the racing tech demo (docs/engine_design.md §18.4), double precision: a rigid body (box inertia)
 * on four spring-damper wheels that sample the ground below their mounts; tire forces with a friction circle
 * (longitudinal: rear wheel drive and brakes, lateral: against the slip, saturating at friction * load).
 * Transitional: Jolt's vehicle constraint replaces it in S28/S29.
 */
class Vehicle
{
   public:
    VehicleParameters params;
    VehicleState state;
    GroundHeight ground;

    // Standing at position (if position(1) is NaN: at the resting height above the ground, the springs compressed by
    // the static load, so the car does not hop), heading (see yawRotation)
    void reset(const Vec3& position, double heading);
    // Height of the center of mass above flat ground when the car rests on its wheels
    double restingHeight() const;
    void step(const VehicleControls& controls, double dt);

    Vec3 forward() const { return forwardOf(state.orientation); }
    Vec3 up() const { return upOf(state.orientation); }
    // Signed speed along the forward axis
    double forwardSpeed() const { return state.velocity.dot(forward()); }
    int wheelsOnGround() const;

   private:
    void subStep(const VehicleControls& c, double dt);
};

}  // namespace Game
}  // namespace Saiga
