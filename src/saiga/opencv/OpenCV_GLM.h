/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "opencv2/opencv.hpp"
#include "saiga/util/math.h"


namespace Saiga
{
template <typename MatrixType>
SAIGA_TEMPLATE inline mat3 CVtoGLM_mat3(const MatrixType& mat)
{
    mat3 M;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            M[j][i] = mat(i, j);
        }
    }
    return M;
}

template <typename MatrixType>
SAIGA_TEMPLATE inline mat4 CVtoGLM_mat4(const MatrixType& mat)
{
    mat4 M;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            M[j][i] = mat(i, j);
        }
    }
    return M;
}


}  // namespace Saiga
