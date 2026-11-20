/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Kinematic character controller of the platformer tech demo.

#include "games/platformer/CharacterController.h"

namespace Saiga
{
namespace Game
{
void CharacterController::reset(const Vec3& position, double yaw)
{
    state          = CharacterState();
    state.position = position;
    state.yaw      = yaw;
}

bool CharacterController::overlaps(const Vec3& p, const LevelBox& b) const
{
    const double eps = 1e-9;
    return p(0) + params.radius > b.min(0) + eps && p(0) - params.radius < b.max(0) - eps &&
           p(2) + params.radius > b.min(2) + eps && p(2) - params.radius < b.max(2) - eps &&
           p(1) + params.height > b.min(1) + eps && p(1) < b.max(1) - eps;
}

bool CharacterController::moveAxis(int axis, double delta)
{
    auto& p = state.position;
    p(axis) += delta;
    bool blocked = false;
    if (!level) return false;
    for (auto& b : level->boxes)
    {
        if (!overlaps(p, b)) continue;
        blocked = true;
        // Push out against the movement
        double lo = axis == 1 ? 0.0 : params.radius;
        double hi = axis == 1 ? params.height : params.radius;
        if (delta > 0)
            p(axis) = b.min(axis) - hi;
        else
            p(axis) = b.max(axis) + lo;
    }
    return blocked;
}

void CharacterController::step(const CharacterInput& in, double dt)
{
    auto& s        = state;
    const auto& p  = params;
    s.landingSpeed = 0;
    s.jumped       = false;

    // Horizontal velocity towards the desired one
    Vec3 move(in.move(0), 0, in.move(2));
    double amount = std::min(move.norm(), 1.0);
    Vec3 dir      = amount > 1e-6 ? Vec3(move.normalized()) : Vec3(0, 0, 0);
    Vec3 desired  = dir * (amount * (in.walk ? p.walkSpeed : p.runSpeed));
    Vec3 horiz(s.velocity(0), 0, s.velocity(2));
    Vec3 diff    = desired - horiz;
    double accel = (s.grounded ? p.groundAcceleration : p.airAcceleration) * dt;
    if (diff.norm() > accel) diff = diff.normalized() * accel;
    s.velocity(0) += diff(0);
    s.velocity(2) += diff(2);

    // Facing
    if (amount > 0.1)
    {
        double target = headingOf(dir);
        s.yaw += wrapAngle(target - s.yaw) * smoothFactor(p.turnRate, dt);
    }

    // Jump (buffered, with coyote time)
    double& jumpBufferTimer = s.jumpBufferTimer;
    jumpBufferTimer         = in.jumpPressed ? p.jumpBuffer : std::max(jumpBufferTimer - dt, 0.0);
    bool canJump            = s.grounded || s.airTime < p.coyoteTime;
    if (jumpBufferTimer > 0 && canJump && s.velocity(1) <= 0.5)
    {
        s.velocity(1)   = p.jumpSpeed;
        s.grounded      = false;
        s.airTime       = p.coyoteTime;
        jumpBufferTimer = 0;
        s.jumped        = true;
    }
    // Gravity, stronger when the jump key is released while rising (variable jump height)
    double g = p.gravity * (s.velocity(1) > 0 && !in.jumpHeld ? 2.0 : 1.0);
    s.velocity(1) -= g * dt;

    // Move in sub steps
    Vec3 delta  = s.velocity * dt;
    int steps   = std::max(1, int(std::ceil(delta.norm() / (p.radius * 0.5))));
    bool landed = false, verticalBlocked = false;
    for (int i = 0; i < steps; ++i)
    {
        if (moveAxis(0, delta(0) / steps)) s.velocity(0) = 0;
        if (moveAxis(2, delta(2) / steps)) s.velocity(2) = 0;
        // Regression: a vertical block ended the loop, which also dropped the remaining horizontal sub steps (half the
        // run speed at 30 Hz); now only the vertical move stops
        if (!verticalBlocked && moveAxis(1, delta(1) / steps))
        {
            if (delta(1) < 0) landed = true;
            // Regression: a blocked upward move kept its speed and the head stuck to the ceiling until gravity used it
            // up
            if (delta(1) > 0) s.velocity(1) = 0;
            verticalBlocked = true;
        }
    }
    bool wasGrounded = s.grounded;
    if (landed)
    {
        if (!wasGrounded) s.landingSpeed = -s.velocity(1);
        s.velocity(1) = 0;
        s.grounded    = true;
        s.airTime     = 0;
    }
    else
    {
        // Still standing? (a probe just below the feet)
        double support = level ? level->supportHeight(s.position(0), s.position(2), s.position(1) + 1e-6) : -1e30;
        bool onTop     = s.velocity(1) <= 0 && s.position(1) - support < 0.02;
        // The box may only touch the support with its corners: check the four corners as well
        if (!onTop && level && s.velocity(1) <= 0)
        {
            for (int c = 0; c < 4 && !onTop; ++c)
            {
                double x = s.position(0) + (c & 1 ? p.radius : -p.radius) * 0.99;
                double z = s.position(2) + (c & 2 ? p.radius : -p.radius) * 0.99;
                onTop    = s.position(1) - level->supportHeight(x, z, s.position(1) + 1e-6) < 0.02;
            }
        }
        s.grounded = onTop;
        if (s.grounded)
            s.airTime = 0;
        else
            s.airTime += dt;
    }
}

}  // namespace Game
}  // namespace Saiga
