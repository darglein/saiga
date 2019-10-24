/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"


namespace Saiga
{
/**
 * The 5 paramter Rad-Tan Distortion model.
 * Stored in the following vector.
 * (k1,k2,p1,p2,k3)
 *
 * If the camera is given as a 4 parameter rad-tan just set k3=0.
 */
template <typename T>
using DistortionBase = Eigen::Matrix<T, 5, 1>;

// Use double by default here
using Distortion  = DistortionBase<double>;
using Distortionf = DistortionBase<float>;


/**
 * The OpenCV distortion model applied to a point in normalized image coordinates.
 */
template <typename T>
Eigen::Matrix<T, 2, 1> distortNormalizedPoint(const Eigen::Matrix<T, 2, 1>& point, const DistortionBase<T>& distortion)
{
    T x  = point.x();
    T y  = point.y();
    T k1 = distortion(0);
    T k2 = distortion(1);
    T p1 = distortion(2);
    T p2 = distortion(3);
    T k3 = distortion(4);

    T x2 = x * x, y2 = y * y;
    T r2 = x2 + y2, _2xy = T(2) * x * y;
    T radial      = (T(1) + ((k3 * r2 + k2) * r2 + k1) * r2);
    T tangentialX = p1 * _2xy + p2 * (r2 + T(2) * x2);
    T tangentialY = p1 * (r2 + T(2) * y2) + p2 * _2xy;
    T xd          = (x * radial + tangentialX);
    T yd          = (y * radial + tangentialY);
    return {xd, yd};
}

/**
 * The inverse OpenCV distortion model with 5 parameters.
 */
template <typename T>
Eigen::Matrix<T, 2, 1> undistortNormalizedPoint(const Eigen::Matrix<T, 2, 1>& point,
                                                const DistortionBase<T>& distortion)
{
    T x  = point.x();
    T y  = point.y();
    T k1 = distortion(0);
    T k2 = distortion(1);
    T p1 = distortion(2);
    T p2 = distortion(3);
    T k3 = distortion(4);

    T x0 = x;
    T y0 = y;
    // compensate distortion iteratively
    constexpr int iters = 5;
    for (int j = 0; j < iters; j++)
    {
        T x2 = x * x, y2 = y * y;
        T r2 = x2 + y2, _2xy = T(2) * x * y;
        T radial      = T(1) / (T(1) + ((k3 * r2 + k2) * r2 + k1) * r2);
        T tangentialX = p1 * _2xy + p2 * (r2 + T(2) * x2);
        T tangentialY = p1 * (r2 + T(2) * y2) + p2 * _2xy;

        x = (x0 - tangentialX) * radial;
        y = (y0 - tangentialY) * radial;
    }
    return {x, y};

}  // namespace Kernel

}  // namespace Saiga
