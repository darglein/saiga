/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"


namespace Saiga
{
template <typename T>
class Triangulation
{
   public:
    using Vec4 = Eigen::Matrix<T, 4, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Mat4 = Eigen::Matrix<T, 4, 4>;
    using SE3  = Sophus::SE3<T>;

    /**
     * Construct a homogeneous linear system of equations:
     * Ax = 0
     * p1,p2 must be in normalized image space!
     *
     * Source:
     * Zisserman Book page p312
     */
    Vec3 triangulateHomogeneous(const SE3& s1, const SE3& s2, const Vec2& p1, const Vec2& p2) const
    {
        Mat4 P1 = s1.matrix();
        Mat4 P2 = s2.matrix();

        Mat4 A;
        A.row(0) = p1.x() * P1.row(2) - P1.row(0);
        A.row(1) = p1.y() * P1.row(2) - P1.row(1);
        A.row(2) = p2.x() * P2.row(2) - P2.row(0);
        A.row(3) = p2.y() * P2.row(2) - P2.row(1);

        auto svd  = A.jacobiSvd(Eigen::ComputeFullV);
        auto V    = svd.matrixV();
        Vec4 phom = V.col(3);
        phom      = phom / phom(3);
        Vec3 p(phom(0), phom(1), phom(2));
        return p;
    }
};



}  // namespace Saiga
