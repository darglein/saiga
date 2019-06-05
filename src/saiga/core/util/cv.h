/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/math/math.h"

namespace Saiga
{
SAIGA_CORE_API mat4 cvCameraToGLCamera(const mat3& K, int viewportW, int viewportH, float znear, float zfar);
SAIGA_CORE_API mat4 cvViewToGLView(const mat4& view);

SAIGA_CORE_API vec2 cvApplyDistortion(vec2 point, float k1, float k2, float k3, float p1, float p2);

}  // namespace Saiga
