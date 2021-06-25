/**
 * Copyright (c) 2021 Darius Rückert
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



inline mat4 CVCamera2GLProjectionMatrix(mat3 K, ivec2 image_size, float znear = .01, float zfar = 1000.)
{
    float fx     = K(0, 0);
    float fy     = K(1, 1);
    float cx     = K(0, 2);
    float cy     = K(1, 2);
    float width  = image_size.x();
    float height = image_size.y();


    mat4 m  = mat4::Zero();
    m(0, 0) = 2.0 * fx / width;
    m(0, 1) = 0.0;
    m(0, 2) = 1.0 - 2.0 * cx / width;
    m(0, 3) = 0.0;

    m(1, 0) = 0.0;
    m(1, 1) = 2.0 * fy / height;
    m(1, 2) = 2.0 * cy / height - 1.0;
    m(1, 3) = 0.0;

    m(2, 0) = 0;
    m(2, 1) = 0;
    m(2, 2) = (zfar + znear) / (znear - zfar);
    m(2, 3) = 2.0 * zfar * znear / (znear - zfar);

    m(3, 0) = 0.0;
    m(3, 1) = 0.0;
    m(3, 2) = -1.0;
    m(3, 3) = 0.0;
    return m;
}

/**
 * In CV the viewport transform is included in the K matrix.
 * The viewport transform is removed (by multiplying the inverse) and
 * a near and far clipping plane is added.
 *
 * The final projection matrix maps a point to the unit cube [-1,1]^3
 */
inline mat3 GLProjectionMatrix2CVCamera(const mat4& proj, int target_w, int target_h)
{
    mat3 K = mat3::Zero();

    K(0, 0) = target_w * proj(0, 0) / 2;
    K(1, 1) = target_h * proj(1, 1) / 2;
    K(2, 2) = 1;

    K(0, 2) = -(proj(0, 2) - 1.0) * target_w / 2.f;
    K(1, 2) = (proj(1, 2) + 1) * target_h / 2.f;


    return K;
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
