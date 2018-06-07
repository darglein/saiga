/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/cv.h"

#include <glm/ext.hpp>

namespace Saiga {


glm::mat4 cvCameraToGLCamera(const glm::mat3& K, int viewportW, int viewportH, float znear, float zfar)
{

    glm::mat3 viewPortTransform(
                viewportW*0.5f,0,viewportW*0.5f,
                0,viewportH*0.5f,viewportH*0.5f,
                0,0,1);
    viewPortTransform = transpose(viewPortTransform);

    auto test = inverse(viewPortTransform) * K;

    mat4 proj(test);
    proj[2][3] = -1;
    proj[3][3] = 0;

    proj[2][2] = -(zfar + znear) / (zfar - znear);
    proj[3][2] = -2.0f * zfar * znear / (zfar - znear);
    return proj;
}

glm::mat4 cvViewToGLView(const glm::mat4 &view)
{
    return glm::rotate(glm::radians(180.0f),vec3(1,0,0)) * view;

}


}
