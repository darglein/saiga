/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

#include "opencv2/opencv.hpp"


namespace Saiga
{
#if 0
template <typename MatrixType>
SAIGA_TEMPLATE inline mat3 CVtoGLM_mat3(const MatrixType& mat)
{
    mat3 M;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            M(i, j) = mat(i, j);
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
            M(i, j) = mat(i, j);
        }
    }
    return M;
}

#endif
}  // namespace Saiga
