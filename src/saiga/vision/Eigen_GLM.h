/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"
#include "saiga/util/glm.h"

namespace Saiga
{

template <typename _Scalar>
SAIGA_TEMPLATE inline mat3 toglm(const Eigen::Matrix<_Scalar,3,3>& mat)
{
    mat3 M;
    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            // Note: glm is col major
            M[j][i] = mat(i, j);
        }
    }
    return M;
}

template <typename _Scalar>
SAIGA_TEMPLATE inline mat4 toglm(const Eigen::Matrix<_Scalar,4,4>& mat)
{
    mat4 M;
    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            // Note: glm is col major
            M[j][i] = mat(i, j);
        }
    }
    return M;
}



}  // namespace Saiga
