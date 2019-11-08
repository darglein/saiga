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
template <typename T>
inline T translationalError(const Sophus::SE3<T>& a, const Sophus::SE3<T>& b)
{
    Vec3 diff = a.translation() - b.translation();
    return diff.norm();
}

// the angle (in radian) between two rotations
template <typename T>
inline T rotationalError(const Sophus::SE3<T>& a, const Sophus::SE3<T>& b)
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
template <typename T>
inline Sophus::SE3<T> scale(const Sophus::SE3<T>& a, double alpha)
{
    return slerp(Sophus::SE3<T>(), a, alpha);
}


/**
 * Construct a skew symmetric matrix from a vector.
 * Also know as 'cross product matrix' or 'hat operator'.
 * https://en.wikipedia.org/wiki/Hat_operator
 */
SAIGA_VISION_API extern Mat3 skew(const Vec3& a);

/**
 * Pixar Revised ONB
 * https://graphics.pixar.com/library/OrthonormalB/paper.pdf
 */
SAIGA_VISION_API extern Mat3 onb(const Vec3& n);

/**
 * Simple ONB from a direction and an up vector.
 */
SAIGA_VISION_API extern Mat3 onb(const Vec3& dir, const Vec3& up);

SAIGA_VISION_API extern Mat3 enforceRank2(const Mat3& M);

SAIGA_VISION_API extern Vec3 infinityVec3();

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
