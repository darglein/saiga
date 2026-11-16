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
/**
 * A third-person / chase camera: follows a target from behind (opposite to its forward direction) and above, smoothed
 * with a frame rate independent spring, kept above the ground. Updated per rendered frame from the interpolated
 * target. Transitional (docs/engine_design.md §18.4): replaced by the engine's camera rig (S35).
 */
class ChaseCamera
{
   public:
    double distance     = 12;
    double height       = 3.5;
    double targetHeight = 1.5;
    // Smoothing rate (1/s) of the eye position
    double stiffness = 5;
    // Minimum height of the eye above the ground (if a ground function is set)
    double groundClearance = 1.0;
    GroundHeight ground;

    // forward: the direction the camera looks along (its xz part is used; length > 0)
    void update(const Vec3& target, const Vec3& forward, double dt);
    // Jumps to the target pose without smoothing
    void snap(const Vec3& target, const Vec3& forward);
    // The next update() snaps (e.g. after a respawn: the camera would glide from the old place otherwise)
    void reset() { initialized = false; }

    const Vec3& eye() const { return eyePos; }
    const Vec3& lookAtPoint() const { return lookPos; }
    // The float view matrix (camera at eye(), looking at lookAtPoint())
    mat4 view() const;

   private:
    Vec3 desiredEye(const Vec3& target, const Vec3& forward) const;
    Vec3 eyePos      = Vec3(0, 0, 0);
    Vec3 lookPos     = Vec3(0, 0, -1);
    bool initialized = false;
};

}  // namespace Game
}  // namespace Saiga
