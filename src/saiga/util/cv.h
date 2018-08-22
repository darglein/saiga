/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/util/glm.h"

namespace Saiga {

SAIGA_GLOBAL glm::mat4 cvCameraToGLCamera(const glm::mat3& K, int viewportW, int viewportH, float znear, float zfar);
SAIGA_GLOBAL glm::mat4 cvViewToGLView(const glm::mat4& view);

SAIGA_GLOBAL vec2 cvApplyDistortion(vec2 point, float k1, float k2 , float k3, float p1, float p2);

}
