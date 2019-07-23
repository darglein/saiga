/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/VisionTypes.h"


namespace Saiga
{
/**
 * Computes the essential matrix from given camera extrinsics.
 * The resulting essential matrix maps points of (a) to lines in (b).
 *
 * 1. Computes relative rotation and translation [R|t] between cameras
 * 2. The essential matrix E is then: E = R[t]x
 */
inline Mat3 EssentialMatrix(const SE3& a, const SE3& b)
{
    SE3 rel = a * b.inverse();
    //    SE3 rel = b * a.inverse();
    return rel.rotationMatrix() * skew(rel.translation());
}


/**
 * Computes the Fundamental Matrix given an essential matrix and
 * the camera intrinsics.
 * Assumes a pinhole camera model!
 *
 * F = K2^-T * E * K1^-1
 */
inline Mat3 FundamentalMatrix(const Mat3& E, const Intrinsics4& K1, const Intrinsics4& K2)
{
    return K2.inverse().matrix().transpose() * E * K1.inverse().matrix();
}

/**
 * Computes the squared distance of point 2 to the epipolar line of point 1.
 */
inline double EpipolarDistanceSquared(const Vec2& p1, const Vec2& p2, const Mat3& F)
{
    Vec3 np1(p1(0), p1(1), 1);
    Vec3 np2(p2(0), p2(1), 1);
    Vec3 l         = F * np1;
    double d       = np2.transpose() * l;
    double lengSqr = l(0) * l(0) + l(1) * l(1);
    double disSqr  = d * d / lengSqr;
    return disSqr;
}

}  // namespace Saiga
