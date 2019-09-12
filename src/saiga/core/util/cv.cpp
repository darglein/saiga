/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/cv.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
mat4 cvCameraToGLCamera(const mat3& K, int viewportW, int viewportH, float znear, float zfar)
{
    /**
     * In CV the viewport transform is included in the K matrix.
     * The viewport transform is removed (by multiplying the inverse) and
     * a near and far clipping plane is added.
     *
     * The final projection matrix maps a point to the unit cube [-1,1]^3
     */
#if 0
    mat3 viewPortTransform(
                0.5f * viewportW,   0,                  0.5f * viewportW,
                0,                  0.5f * viewportH,   0.5f * viewportH,
                0,                  0,                  1
                );
    auto removeViewPortTransform = inverse(transpose(viewPortTransform));
    std::cout << viewPortTransform << std::endl << removeViewPortTransform << std::endl;
#else
    mat3 removeViewPortTransform = make_mat3(2.0 / viewportW, 0, 0, 0, 2.0 / viewportH, 0, -1, -1, 1);
#endif
    auto test = removeViewPortTransform * K;

    mat4 proj       = make_mat4(test);
    col(proj, 2)[3] = -1;
    col(proj, 3)[3] = 0;

    col(proj, 2)[2] = -(zfar + znear) / (zfar - znear);
    col(proj, 3)[2] = -2.0f * zfar * znear / (zfar - znear);
    return proj;
}

mat4 cvViewToGLView(const mat4& view)
{
    /**
     * In computer vision the y-axis points down and the looks in the positive z-direction.
     *
     * Both systems are right-handed.
     */
    mat4 viewTransform = make_mat4(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);
    return viewTransform * view;
}

vec2 cvApplyDistortion(vec2 point, float k1, float k2, float k3, float p1, float p2)
{
    /**
     * The OpenCV distortion model applied to a point in normalized image coordinates.
     */
    using T = float;
    T x     = point[0];
    T y     = point[1];
    T x2 = x * x, y2 = y * y;
    T r2 = x2 + y2, _2xy = T(2) * x * y;
    T radial      = (T(1) + ((k3 * r2 + k2) * r2 + k1) * r2);
    T tangentialX = p1 * _2xy + p2 * (r2 + T(2) * x2);
    T tangentialY = p1 * (r2 + T(2) * y2) + p2 * _2xy;
    T xd          = (x * radial + tangentialX);
    T yd          = (y * radial + tangentialY);
    return vec2(xd, yd);
}


}  // namespace Saiga
