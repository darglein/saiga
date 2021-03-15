/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "math.h"

/**
 * Important Note:
 *
 * Saiga uses OpenGL Projection and View Matrices for Rendering, Shaders, Controllable cameras, etc.
 * Make sure to convert it to the GL system using the function below.
 *
 * For example, if you have a Computer Vision pose (world -> camera transformation) you want to use the function
 * CV2GLView() before setting it to a camera:
 *
 * camera.view = CV2GLView() * cv_pose;
 */


/**
 * Projection Matrix Differences
 *
 * OpenGL: Left handed normalized image coordinates with:
 *      x: right    Range [-1,1]
 *      y: up       Range [-1,1]
 *      z: forward  Range [-1,1]
 *
 * Vulkan: Right handed normalized image coordinate with:
 *      x: right    Range [-1,1]
 *      y: down     Range [-1,1]
 *      z: forward  Range [0,1]
 *
 * CV: Right handed but without range restriction (the clipping is done when tranforming to image space):
 *      x: right
 *      y: down
 *      z: forward
 */


/**
 * View Matrix Differences
 *
 * OpenGL: Right handed with:
 *      x: right
 *      y: up
 *      z: backwards (negative z-values are in front of the camera)
 *
 * Computer Vision (CV): Right handed with:
 *      x: right
 *      y: down
 *      z: forward (negative z-values are behind the camera)
 */

namespace Saiga
{
/**
 * To get from OpenGL to Vulkan we have to invert the y-axis and
 * transform the z range from [-1,1] to [0,1].
 */
inline mat4 GL2VulkanNormalizedImage()
{
    // clang-format off
    return make_mat4_row_major(
        1.0f,  0.0f,  0.0f,  0.0f,
        0.0f, -1.0f,  0.0f,  0.0f,
        0.0f,  0.0f,  0.5f,  0.5f,
        0.0f,  0.0f,  0.0f,  1.0f
    );
    // clang-format on
}

/**
 * Same as above but inverse
 */
inline mat4 Vulkan2GLNormalizedImage()
{
    // clang-format off
    return make_mat4_row_major(
        1.0f,  0.0f,  0.0f,  0.0f,
        0.0f, -1.0f,  0.0f,  0.0f,
        0.0f,  0.0f,  2.0f, -1.0f,
        0.0f,  0.0f,  0.0f,  1.0f
    );
    // clang-format on
}


/**
 * To get from OpenGL to CV we have to flip the y and z axis.
 */
inline mat4 GL2CVView()
{
    // clang-format off
    return make_mat4_row_major(
        1.0f,  0.0f,  0.0f,  0.0f,
        0.0f, -1.0f,  0.0f,  0.0f,
        0.0f,  0.0f, -1.0f,  0.0f,
        0.0f,  0.0f,  0.0f,  1.0f
    );
    // clang-format on
}

/**
 * The inverse of the above is itself.
 */
inline mat4 CV2GLView()
{
    return GL2CVView();
}


inline mat4 cvCameraToGLProjectionMatrix(const mat3& K, int viewportW, int viewportH, float znear, float zfar)
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

    mat4 proj = make_mat4(test);
    //    col(proj, 2)[3] = -1;
    //    col(proj, 3)[3] = 0;
    //    col(proj, 2)[2] = -(zfar + znear) / (zfar - znear);
    //    col(proj, 3)[2] = -2.0f * zfar * znear / (zfar - znear);
    proj(3, 2) = -1;
    proj(3, 3) = 0;
    proj(2, 2) = -(zfar + znear) / (zfar - znear);
    proj(2, 3) = -2.0f * zfar * znear / (zfar - znear);
    return proj;
}


inline vec2 cvApplyDistortion(vec2 point, float k1, float k2, float k3, float p1, float p2)
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
