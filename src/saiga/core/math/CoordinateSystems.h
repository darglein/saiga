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

}  // namespace Saiga
