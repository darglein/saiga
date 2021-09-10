/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/cameraModel/Distortion.h"
#include "saiga/vision/cameraModel/Intrinsics4.h"
#include "saiga/vision/cameraModel/Rectify.h"

namespace Saiga
{


/**
 * Construct a skew symmetric matrix from a vector.
 * Also know as 'cross product matrix' or 'hat operator'.
 * https://en.wikipedia.org/wiki/Hat_operator
 *
 * Vector [x,y,z] transforms to Matrix
 *
 * |  0  -z   y |
 * |  z   0  -x |
 * | -y   x   0 |
 *
 */

template <typename T>
HD inline Matrix<T, 3, 3> skew(const Vector<T, 3>& a)
{
    Matrix<T, 3, 3> m;
    // clang-format off
    m <<
        0,      -a(2),  a(1),
        a(2),   0,      -a(0),
        -a(1),  a(0),   0;
    // clang-format on

    return m;
}



SAIGA_VISION_API extern Mat3 enforceRank2(const Mat3& M);

SAIGA_VISION_API extern Vec3 infinityVec3();


}  // namespace Saiga


namespace Eigen
{
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
}  // namespace Eigen

namespace Sophus
{
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Sophus::SE3<T>& se3)
{
    auto q = se3.unit_quaternion();
    auto t = se3.translation();
    os << "SE3(Quatwxyz(" << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "),Vec3(" << t(0) << "," << t(1)
       << "," << t(2) << "))";
    return os;
}


// template <typename T>
// inline std::ostream& operator<<(std::ostream& os, const Sophus::Sim3<T>& sim3)
//{
//    auto se3_scale = Saiga::se3Scale(sim3);

//    os << "Sim3(" << se3_scale.first << " Scale=" << se3_scale.second << ")";
//    return os;
//}

template <typename T>
inline bool operator==(const Sophus::SE3<T>& a, const Sophus::SE3<T>& b)
{
    return a.params() == b.params();
}



}  // namespace Sophus
