/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VisionTypes.h"

#include "VisionIncludes.h"

namespace Saiga
{
Mat3 skew(const Vec3& a)
{
    Mat3 m;
    using Scalar = double;
    m << Scalar(0), -a(2), a(1), a(2), Scalar(0), -a(0), -a(1), a(0), Scalar(0);
    return m;
}

Mat3 onb(const Vec3& n)
{
    double sign = n(2) > 0 ? 1.0f : -1.0f;  // emulate copysign
    double a    = -1.0f / (sign + n[2]);
    double b    = n[0] * n[1] * a;
    Mat3 v;
    v.col(2) = n;
    v.col(1) = Vec3(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
    v.col(0) = Vec3(b, sign + n[1] * n[1] * a, -n[1]);
    return v;
}

Mat3 onb(const Vec3& dir, const Vec3& up)
{
    Mat3 R;
    R.col(2) = dir.normalized();
    R.col(1) = up.normalized();
    R.col(0) = R.col(1).cross(R.col(2)).normalized();
    // make sure it works even if dir and up are not orthogonal
    R.col(1) = R.col(2).cross(R.col(0));
    return R;
}

Mat3 enforceRank2(const Mat3& M)
{
    // enforce it with svd
    // det(F)=0
    auto svde = M.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto svs  = svde.singularValues();
    // set third singular value to 0
    svs(2) = 0;

    Eigen::DiagonalMatrix<double, 3> sigma;
    sigma.diagonal() = svs;

    Mat3 u = svde.matrixU();
    Mat3 v = svde.matrixV();
    auto F = u * sigma * v.transpose();
    return F;
}

Vec3 infinityVec3()
{
    return Vec3(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity());
}

}  // namespace Saiga
