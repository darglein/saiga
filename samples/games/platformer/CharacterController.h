/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/platformer/Level.h"

#include <string>

namespace Saiga
{
namespace Game
{
struct CharacterInput
{
    // Desired horizontal direction (world xz, length <= 1)
    Vec3 move = Vec3(0, 0, 0);
    bool walk = false;
    // Jump key held (variable jump height) and pressed this step
    bool jumpHeld = false, jumpPressed = false;
};

// The data file data/platformer/character_parameters.json (reload_data tunes it live)
struct CharacterParameters
{
    static constexpr const char* name = "CharacterParameters";
    static constexpr uint32_t version = 1;

    // Axis aligned collision box around the feet position: half width and height
    double radius = 0.35, height = 1.8;
    double gravity = 25, jumpSpeed = 9;
    double runSpeed = 6.5, walkSpeed = 2.5;
    double groundAcceleration = 45, airAcceleration = 14;
    // Jumping shortly after leaving a ledge (coyote time) and shortly before landing (buffer)
    double coyoteTime = 0.1, jumpBuffer = 0.12;
    // Turning towards the movement direction (1/s)
    double turnRate = 12;

    template <class V>
    void visit(V& v)
    {
        v("radius", radius);
        v("height", height);
        v("gravity", gravity);
        v("jumpSpeed", jumpSpeed);
        v("runSpeed", runSpeed);
        v("walkSpeed", walkSpeed);
        v("groundAcceleration", groundAcceleration);
        v("airAcceleration", airAcceleration);
        v("coyoteTime", coyoteTime);
        v("jumpBuffer", jumpBuffer);
        v("turnRate", turnRate);
    }
    // Checked when the data file is read (load and reload_data): the controller divides by the radius
    bool validate(std::string& error) const
    {
        if (!(radius >= 0.01 && height > 0 && gravity >= 0 && jumpSpeed >= 0 && runSpeed >= 0 && walkSpeed >= 0 &&
              groundAcceleration >= 0 && airAcceleration >= 0 && coyoteTime >= 0 && jumpBuffer >= 0 && turnRate >= 0))
        {
            error = "radius >= 0.01, height > 0, and the speeds, accelerations, times and rates >= 0";
            return false;
        }
        return true;
    }
};

struct CharacterState
{
    // Feet position (center of the bottom of the collision box)
    Vec3 position = Vec3(0, 0, 0);
    Vec3 velocity = Vec3(0, 0, 0);
    // Facing (see yawRotation)
    double yaw     = 0;
    bool grounded  = false;
    double airTime = 0;
    // Set in the step that landed: the vertical speed at impact (m/s, positive)
    double landingSpeed = 0;
    bool jumped         = false;
    // Time left of a buffered jump press
    double jumpBufferTimer = 0;
};

/**
 * Kinematic character controller of the platformer tech demo (docs/engine_design.md §18.4): gravity, jump with coyote
 * time and buffering, acceleration on the ground and in the air, collision of an axis aligned box with the level boxes
 * (moved axis by axis in sub steps no longer than half the radius, so thin platforms are not tunnelled through).
 * Transitional: Jolt's CharacterVirtual replaces it in S29.
 */
class CharacterController
{
   public:
    CharacterParameters params;
    CharacterState state;
    const Level* level = nullptr;

    void reset(const Vec3& position, double yaw = 0);
    void step(const CharacterInput& input, double dt);
    double horizontalSpeed() const { return Vec3(state.velocity(0), 0, state.velocity(2)).norm(); }

   private:
    // Moves along one axis and resolves the collisions; returns true if blocked
    bool moveAxis(int axis, double delta);
    bool overlaps(const Vec3& p, const LevelBox& b) const;
};

}  // namespace Game
}  // namespace Saiga
