/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/HomogeneousLSE.h"
#include "saiga/vision/VisionTypes.h"


namespace Saiga
{
/**
 * Construct a homogeneous linear system of equations:
 * Ax = 0
 * p1,p2 must be in normalized image space!
 *
 * Source:
 * Zisserman Book page p312
 */
template <typename T, bool Precise = true>
Vec3 TriangulateHomogeneous(const SE3& s1, const SE3& s2, const Vec2& p1, const Vec2& p2)
{
    using Vec4 = Eigen::Matrix<T, 4, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    //    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Mat4 = Eigen::Matrix<T, 4, 4>;
    //    using SE3  = Sophus::SE3<T>;

    Mat4 P1 = s1.matrix();
    Mat4 P2 = s2.matrix();

    Mat4 A;
    A.row(0) = p1.x() * P1.row(2) - P1.row(0);
    A.row(1) = p1.y() * P1.row(2) - P1.row(1);
    A.row(2) = p2.x() * P2.row(2) - P2.row(0);
    A.row(3) = p2.y() * P2.row(2) - P2.row(1);

    Vec4 phom;

    // svd is a bit more precise, but also a lot slower
    if constexpr (Precise)
        solveHomogeneousJacobiSVD(A, phom);
    else
        solveHomogeneousQR(A, phom);

    phom = phom / phom(3);
    Vec3 p(phom(0), phom(1), phom(2));
    return p;
}



// Compute the triangulation angle in radians
// c1: world position of camera 1
// c2: world position of camera 2
// wp: world position of 3D point
// This code here is partially copied from Colmap.
// https://github.com/colmap/colmap
inline double TriangulationAngle(const Vec3& c1, const Vec3& c2, const Vec3& wp)
{
#if 0
    // alternative implementation using the dot product
    // (requires 2 sqrts so is a bit slower)
    Vec3 v1      = (c1 - wp).normalized();
    Vec3 v2      = (c2 - wp).normalized();
    double cosA  = v1.dot(v2);
    double angle = acos(cosA);
#else
    // Baseline length between camera centers.
    const double baseline_length_squared = (c1 - c2).squaredNorm();

    // Ray lengths from cameras to point.
    const double ray_length_squared1 = (wp - c1).squaredNorm();
    const double ray_length_squared2 = (wp - c2).squaredNorm();

    // Using "law of cosines" to compute the enclosing angle between rays.
    const double denominator = 2.0 * std::sqrt(ray_length_squared1 * ray_length_squared2);
    if (denominator == 0.0)
    {
        return 0.0;
    }
    const double nominator = ray_length_squared1 + ray_length_squared2 - baseline_length_squared;
    const double angle     = std::abs(std::acos(nominator / denominator));
#endif

    // Triangulation is unstable for acute angles (far away points) and
    // obtuse angles (close points), so always compute the minimum angle
    // between the two intersecting rays.
    return std::min(angle, pi<double>() - angle);
}


}  // namespace Saiga
