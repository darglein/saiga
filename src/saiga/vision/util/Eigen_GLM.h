/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"



#include "saiga/vision/VisionIncludes.h"

namespace Saiga
{
/**
 * Don't do anything because the mat3/4 types are already eigen
 */
template <typename _Scalar>
SAIGA_TEMPLATE inline mat3 toglm(const Eigen::Matrix<_Scalar, 3, 3>& mat)
{
    return mat.template cast<float>();
}

template <typename _Scalar>
SAIGA_TEMPLATE inline mat4 toglm(const Eigen::Matrix<_Scalar, 4, 4>& mat)
{
    return mat.template cast<float>();
}


template <typename _Scalar>
SAIGA_TEMPLATE inline vec3 toglm(const Eigen::Matrix<_Scalar, 3, 1>& v)
{
    return v.template cast<float>();
}


}  // namespace Saiga
