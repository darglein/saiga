/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"



#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sophus/se3.hpp"


namespace Saiga {


using SE3 = Sophus::SE3d;

using Quat = Eigen::Quaterniond;

using Vec3 = Eigen::Vector3d;
using Vec2 = Eigen::Vector2d;

using Mat4 = Eigen::Matrix4d;
using Mat3 = Eigen::Matrix3d;


struct Intrinsics4
{
    double fx, fy;
    double cx, cy;

    Intrinsics4(){}
    Intrinsics4(double fx, double fy, double cx, double cy) :
        fx(fx),fy(fy),cx(cx),cy(cy) {}

    Vec2 project(const Vec3& X) const
    {
        auto x = X(0) / X(2);
        auto y = X(1) / X(2);
        return {fx * x + cx, fy * y + cy };
    }

    Vec3 project3(const Vec3& X) const
    {
        auto x = X(0) / X(2);
        auto y = X(1) / X(2);
        return {fx * x + cx, fy * y + cy, X(2) };
    }

    Vec3 unproject(const Vec2& ip, double depth) const
    {
        Vec3 p( (ip(0)-cx) / fx,
                (ip(1)-cy) / fy,
                1);
        return p * depth;
    }
};


inline Vec3 infinityVec3() { return Vec3(std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity());}

inline
double translationalError(const SE3& a, const SE3& b)
{
    Vec3 diff = a.translation() - b.translation();
    return diff.norm();
}

// the angle (in radian) between two rotations
inline
double rotationalError(const SE3& a, const SE3& b)
{
    Quat q1 = a.unit_quaternion();
    Quat q2 = b.unit_quaternion();
    return q1.angularDistance(q2);
}

// the angle (in radian) between two rotations
inline
SE3 slerp(const SE3& a, const SE3& b, double alpha)
{
    Vec3 t = (1.0 -  alpha) * a.translation() + (alpha) * b.translation();

    Quat q1 = a.unit_quaternion();
    Quat q2 = b.unit_quaternion();
    Quat q = q1.slerp(alpha,q2);

    return SE3(q,t);
}

inline
std::ostream& operator<<(std::ostream& os, const Saiga::SE3& se3)
{
    os << se3.translation().transpose() << " | " << se3.unit_quaternion().coeffs().transpose();
    return os;
}

}
