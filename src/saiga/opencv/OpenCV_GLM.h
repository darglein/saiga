/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "opencv2/opencv.hpp"
#include "saiga/util/glm.h"


namespace Saiga
{

template<typename MatrixType>
SAIGA_TEMPLATE inline
glm::mat3 CVtoGLM_mat3(const MatrixType& mat)
{
    glm::mat3 M;
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            M[j][i] = mat(i,j);
        }
    }
    return M;
}

template<typename MatrixType>
SAIGA_TEMPLATE inline
glm::mat4 CVtoGLM_mat4(const MatrixType& mat)
{
    glm::mat4 M;
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            M[j][i] = mat(i,j);
        }
    }
    return M;
}


}
