/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/vision/cameraModel/Intrinsics4.h"

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

/**
 * Undistorts all points from begin to end and writes them to output.
 */
template <typename _InputIterator1, typename _InputIterator2, typename _T>
inline void undistortAll(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __output,
                         const Intrinsics4Base<_T>& intr, const DistortionBase<_T>& dis)
{
    for (; __first1 != __last1; ++__first1, (void)++__output)
    {
        auto tmp  = *__first1;  // make sure it works inplace
        tmp       = intr.unproject2(tmp);
        tmp       = undistortNormalizedPoint(tmp, dis);
        tmp       = intr.normalizedToImage(tmp);
        *__output = tmp;
    }
}

}  // namespace Saiga
