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

SAIGA_GLOBAL inline
glm::mat3 CVtoGLM_mat3(cv::Mat1d mat)
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

SAIGA_GLOBAL inline
glm::mat4 CVtoGLM_mat4(cv::Mat1d mat)
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
