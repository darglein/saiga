/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
template <typename T>
Sophus::SE3<T> convertSE3(const SE3& se3)
{
    Sophus::SE3<T> result;
    for (int i = 0; i < 3; ++i) result.translation()(i) = T(se3.translation()(i));

    for (int i = 0; i < 4; ++i) result.so3().data()[i] = T(se3.so3().data()[i]);

    return result;
}


template <typename T>
Eigen::Matrix<T, 3, 1> convertVec(const Vec3& v)
{
    Eigen::Matrix<T, 3, 1> result;
    for (int i = 0; i < 3; ++i) result(i) = T(v(i));
    return result;
}


}  // namespace Saiga
