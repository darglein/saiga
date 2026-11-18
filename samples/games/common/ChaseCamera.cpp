/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Smoothed chase / third-person camera of the tech demo games.

#include "games/common/ChaseCamera.h"

namespace Saiga
{
namespace Game
{
Vec3 ChaseCamera::desiredEye(const Vec3& target, const Vec3& forward) const
{
    Vec3 f(forward(0), 0, forward(2));
    if (f.norm() < 1e-6) f = Vec3(0, 0, -1);
    f.normalize();
    Vec3 e = target - f * distance + Vec3(0, height, 0);
    if (ground) e(1) = std::max(e(1), ground(e(0), e(2)) + groundClearance);
    return e;
}

void ChaseCamera::snap(const Vec3& target, const Vec3& forward)
{
    // desiredEye() keeps the eye above the ground
    eyePos      = desiredEye(target, forward);
    lookPos     = target + Vec3(0, targetHeight, 0);
    initialized = true;
}

void ChaseCamera::update(const Vec3& target, const Vec3& forward, double dt)
{
    if (!initialized)
    {
        snap(target, forward);
        return;
    }
    Vec3 desired = desiredEye(target, forward);
    eyePos += (desired - eyePos) * smoothFactor(stiffness, dt);
    if (ground) eyePos(1) = std::max(eyePos(1), ground(eyePos(0), eyePos(2)) + groundClearance);
    lookPos = target + Vec3(0, targetHeight, 0);
}

mat4 ChaseCamera::view() const
{
    vec3 e = eyePos.cast<float>(), c = lookPos.cast<float>();
    if ((c - e).norm() < 1e-4f) c = e + vec3(0, 0, -1);
    return lookAt(e, c, vec3(0, 1, 0));
}

}  // namespace Game
}  // namespace Saiga
