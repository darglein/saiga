/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

#include <algorithm>
#include <cmath>
#include <functional>

/**
 * Small double precision helpers of the tech demo games (docs/engine_design.md §18.4). The simulation state of the
 * games is double (R-MATH-1), rendering converts to float relative to the camera.
 */
namespace Saiga
{
namespace Game
{
constexpr double kPi = 3.14159265358979323846;

// The ground height at (x, z). The games get it from the world prototype's height field.
using GroundHeight = std::function<double(double x, double z)>;

// Rotation by angle (radians) around axis (any length > 0)
inline Quat axisAngle(const Vec3& axis, double angle)
{
    Vec3 a   = axis.normalized();
    double s = std::sin(angle * 0.5);
    return Quat(std::cos(angle * 0.5), a(0) * s, a(1) * s, a(2) * s);
}

// Rotation around +y. Heading 0 looks along -z (the forward direction of all game objects), positive turns left.
inline Quat yawRotation(double yaw)
{
    return axisAngle(Vec3(0, 1, 0), yaw);
}

// The forward (-z), up (+y) and right (+x) axes of an orientation
inline Vec3 forwardOf(const Quat& q)
{
    return q * Vec3(0, 0, -1);
}
inline Vec3 upOf(const Quat& q)
{
    return q * Vec3(0, 1, 0);
}
inline Vec3 rightOf(const Quat& q)
{
    return q * Vec3(1, 0, 0);
}

// q rotated by the angular velocity omega (rad/s, world frame) for dt seconds
inline Quat integrateWorld(const Quat& q, const Vec3& omega, double dt)
{
    double angle = omega.norm() * dt;
    if (angle < 1e-12) return q;
    return (axisAngle(omega, angle) * q).normalized();
}

// q rotated by the angular velocity omega (rad/s, body frame) for dt seconds
inline Quat integrateBody(const Quat& q, const Vec3& omega, double dt)
{
    double angle = omega.norm() * dt;
    if (angle < 1e-12) return q;
    return (q * axisAngle(omega, angle)).normalized();
}

// Translation * rotation * scale
inline Mat4 transformMatrix(const Vec3& t, const Quat& r, const Vec3& s = Vec3(1, 1, 1))
{
    Mat3 R = r.matrix();
    Mat4 m = Mat4::Identity();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j) m(i, j) = R(i, j) * s(j);
        m(i, 3) = t(i);
    }
    return m;
}

inline Vec3 transformPoint(const Mat4& m, const Vec3& p)
{
    Vec3 r;
    for (int i = 0; i < 3; ++i) r(i) = m(i, 0) * p(0) + m(i, 1) * p(1) + m(i, 2) * p(2) + m(i, 3);
    return r;
}

// Frame rate independent exponential smoothing: the fraction to move towards the target in dt at the given rate (1/s)
inline double smoothFactor(double rate, double dt)
{
    return 1.0 - std::exp(-rate * dt);
}

inline double clampd(double v, double lo, double hi)
{
    return std::min(std::max(v, lo), hi);
}

// The angle wrapped into [-pi, pi]
inline double wrapAngle(double a)
{
    if (!std::isfinite(a)) return 0;
    a = std::fmod(a, 2 * kPi);
    if (a > kPi) a -= 2 * kPi;
    if (a < -kPi) a += 2 * kPi;
    return a;
}

// Signed angle from a to b in the xz plane (positive: b is to the left of a, i.e. a counter clockwise turn seen from
// above with heading 0 along -z)
inline double signedAngleXZ(const Vec3& a, const Vec3& b)
{
    double ha = std::atan2(-a(0), -a(2));
    double hb = std::atan2(-b(0), -b(2));
    return wrapAngle(hb - ha);
}

// The heading (yaw) of a direction in the xz plane, see yawRotation
inline double headingOf(const Vec3& dir)
{
    return std::atan2(-dir(0), -dir(2));
}

// Normal of the ground from central differences
inline Vec3 groundNormal(const GroundHeight& ground, double x, double z, double eps = 1.0)
{
    double dx = ground(x + eps, z) - ground(x - eps, z);
    double dz = ground(x, z + eps) - ground(x, z - eps);
    return Vec3(-dx, 2 * eps, -dz).normalized();
}

// Normalized weighted average of quaternions (sign aligned to the first one); the weights need not sum to 1
inline Quat blendRotations(const Quat* q, const double* w, int n)
{
    Vec4 sum = Vec4::Zero();
    for (int i = 0; i < n; ++i)
    {
        Vec4 c = q[i].coeffs();
        if (i > 0 && c.dot(q[0].coeffs()) < 0) c = -c;
        sum += c * w[i];
    }
    double l = sum.norm();
    if (l < 1e-12) return Quat::Identity();
    sum /= l;
    return Quat(sum(3), sum(0), sum(1), sum(2));
}

}  // namespace Game
}  // namespace Saiga
