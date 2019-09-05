/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/cameraModel/Distortion.h"
#include "saiga/vision/cameraModel/Intrinsics4.h"

namespace Saiga
{
inline Vec3 reprojectionError(const Vec3& observed, const Vec3& p)
{
    return {observed(0) - p(0), observed(1) - p(1), 0};
}

inline Vec3 reprojectionErrorDepth(const Vec3& observed, const Vec3& p, double bf)
{
    double stereoPointObs = observed(0) - bf / observed(2);
    double stereoPoint    = p(0) - bf / p(2);
    return {observed(0) - p(0), observed(1) - p(1), stereoPointObs - stereoPoint};
}


inline Vec3 infinityVec3()
{
    return Vec3(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity());
}

inline double translationalError(const SE3& a, const SE3& b)
{
    Vec3 diff = a.translation() - b.translation();
    return diff.norm();
}

// the angle (in radian) between two rotations
inline double rotationalError(const SE3& a, const SE3& b)
{
    Quat q1 = a.unit_quaternion();
    Quat q2 = b.unit_quaternion();
    return q1.angularDistance(q2);
}

// Spherical interpolation
template <typename T>
inline Sophus::SE3<T> slerp(const Sophus::SE3<T>& a, const Sophus::SE3<T>& b, T alpha)
{
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Quat = Eigen::Quaternion<T>;

    Vec3 t  = (1.0 - alpha) * a.translation() + (alpha)*b.translation();
    Quat q1 = a.unit_quaternion();
    Quat q2 = b.unit_quaternion();
    Quat q  = q1.slerp(alpha, q2);
    return Sophus::SE3<T>(q, t);
}

// scale the transformation by a scalar
inline SE3 scale(const SE3& a, double alpha)
{
    return slerp(SE3(), a, alpha);
}


/**
 * Construct a skew symmetric matrix from a vector.
 * Also know as 'cross product matrix' or 'hat operator'.
 * https://en.wikipedia.org/wiki/Hat_operator
 */
inline Mat3 skew(Vec3 const& a)
{
    Mat3 m;
    using Scalar = double;
    m << Scalar(0), -a(2), a(1), a(2), Scalar(0), -a(0), -a(1), a(0), Scalar(0);
    return m;
}

/**
 * Pixar Revised ONB
 * https://graphics.pixar.com/library/OrthonormalB/paper.pdf
 */
inline Mat3 onb(const Vec3& n)
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

/**
 * Simple ONB from a direction and an up vector.
 */
inline Mat3 onb(const Vec3& dir, const Vec3& up)
{
    Mat3 R;
    R.col(2) = dir.normalized();
    R.col(1) = up.normalized();
    R.col(0) = R.col(1).cross(R.col(2)).normalized();
    // make sure it works even if dir and up are not orthogonal
    R.col(1) = R.col(2).cross(R.col(0));
    return R;
}

inline Mat3 enforceRank2(const Mat3& M)
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


/**
 * Undistorts all points from begin to end and writes them to output.
 */
template <typename _InputIterator1, typename _InputIterator2, typename _T>
inline void undistortAll(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __output,
                         const Intrinsics4Base<_T>& intr, const DistortionBase<_T>& dis)
{
    for (; __first1 != __last1; ++__first1, (void)++__output)
    {
        auto tmp  = *__first1;  // make sure it works inplace
        tmp       = intr.unproject2(tmp);
        tmp       = undistortNormalizedPoint(tmp, dis);
        tmp       = intr.normalizedToImage(tmp);
        *__output = tmp;
    }
}



template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Sophus::SE3<T>& se3)
{
    Quat q = se3.unit_quaternion();
    Vec3 t = se3.translation();
    os << "SE3(Quat(" << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "),Vec3(" << t(0) << "," << t(1)
       << "," << t(2) << "))";

    return os;
}


template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Sophus::Sim3<T>& sim3)
{
    Quat q = sim3.rxso3().quaternion();
    Vec3 t = sim3.translation();
    os << "Sim3(ScaledQuat(" << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "),Vec3(" << t(0) << ","
       << t(1) << "," << t(2) << "))";

    return os;
}



template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline std::istream& operator>>(std::istream& is, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& m)
{
    for (int i = 0; i < m.rows(); ++i)
    {
        for (int j = 0; j < m.cols(); ++j)
        {
            is >> m(i, j);
        }
    }
    return is;
}

}  // namespace Saiga
