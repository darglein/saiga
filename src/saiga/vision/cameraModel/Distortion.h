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
struct DistortionBase
{
    T k1 = 0;
    T k2 = 0;
    T k3 = 0;
    T k4 = 0;
    T k5 = 0;
    T k6 = 0;
    T p1 = 0;
    T p2 = 0;

    DistortionBase() {}

    DistortionBase(const Eigen::Matrix<T, 8, 1>& c)
    {
        k1 = c(0);
        k2 = c(1);
        k3 = c(2);
        k4 = c(3);
        k5 = c(4);
        k6 = c(5);
        p1 = c(6);
        p2 = c(7);
    }

    Eigen::Matrix<T, 8, 1> Coeffs() const
    {
        Eigen::Matrix<T, 8, 1> result;
        result << k1, k2, k3, k4, k5, k6, p1, p2;
        return result;
    }


    Eigen::Matrix<T, 8, 1> OpenCVOrder()
    {
        Eigen::Matrix<T, 8, 1> result;
        result << k1, k2, p1, p2, k3, k4, k5, k6;
        return result;
    }
};

// Use double by default here
using Distortion  = DistortionBase<double>;
using Distortionf = DistortionBase<float>;



/**
 * The OpenCV distortion model applied to a point in normalized image coordinates.
 */


template <typename T>
Eigen::Matrix<T, 2, 1> distortNormalizedPoint(const Eigen::Matrix<T, 2, 1>& point, const DistortionBase<T>& distortion,
                                              Matrix<double, 2, 2>* J_point = nullptr)
{
    T x  = point.x();
    T y  = point.y();
    T k1 = distortion.k1;
    T k2 = distortion.k2;
    T k3 = distortion.k3;

    T k4 = distortion.k4;
    T k5 = distortion.k5;
    T k6 = distortion.k6;

    T p1 = distortion.p1;
    T p2 = distortion.p2;

    T x2 = x * x, y2 = y * y;
    T r2 = x2 + y2, _2xy = T(2) * x * y;

    T radial_u = T(1) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    T radial_v = T(1) + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2;
    T radial   = (radial_u / radial_v);

    T tangentialX = p1 * _2xy + p2 * (r2 + T(2) * x2);
    T tangentialY = p1 * (r2 + T(2) * y2) + p2 * _2xy;

    T xd = x * radial + tangentialX;
    T yd = y * radial + tangentialY;

    if (J_point)
    {
        auto& Jp = *J_point;
        Jp.setZero();


        Mat2 J_rad_u;
        J_rad_u(0, 0) = k1 * (2 * x);
        J_rad_u(0, 1) = k1 * (2 * y);
        J_rad_u(0, 0) += k2 * (4 * x * x * x + 4 * x * y * y);
        J_rad_u(0, 1) += k2 * (4 * y * y * y + 4 * y * x * x);
        J_rad_u(0, 0) += k3 * (6 * x2 * x2 * x + 12 * x2 * x * y2 + 6 * x * y2 * y2);
        J_rad_u(0, 1) += k3 * (6 * y2 * y2 * y + 12 * y2 * y * x2 + 6 * y * x2 * x2);
        J_rad_u(1, 0) = J_rad_u(0, 0);
        J_rad_u(1, 1) = J_rad_u(0, 1);


        Mat2 J_rad_v;
        J_rad_v(0, 0) = k4 * (2 * x);
        J_rad_v(0, 1) = k4 * (2 * y);
        J_rad_v(0, 0) += k5 * (4 * x * x * x + 4 * x * y * y);
        J_rad_v(0, 1) += k5 * (4 * y * y * y + 4 * y * x * x);
        J_rad_v(0, 0) += k6 * (6 * x2 * x2 * x + 12 * x2 * x * y2 + 6 * x * y2 * y2);
        J_rad_v(0, 1) += k6 * (6 * y2 * y2 * y + 12 * y2 * y * x2 + 6 * y * x2 * x2);
        J_rad_v(1, 0) = J_rad_v(0, 0);
        J_rad_v(1, 1) = J_rad_v(0, 1);

        Mat2 J_rad;
        J_rad = (J_rad_u * radial_v - J_rad_v * radial_u) / (radial_v * radial_v);

        Mat2 J_rad_mul_xy;
        J_rad_mul_xy(0, 0) = x * J_rad(0, 0) + radial;
        J_rad_mul_xy(0, 1) = x * J_rad(0, 1);
        J_rad_mul_xy(1, 0) = y * J_rad(1, 0);
        J_rad_mul_xy(1, 1) = y * J_rad(1, 1) + radial;

        Mat2 J_tan;
        J_tan(0, 0) = 2 * p1 * y + 6 * p2 * x;
        J_tan(0, 1) = 2 * p1 * x + 2 * p2 * y;

        J_tan(1, 0) = 2 * p2 * y + 2 * p1 * x;
        J_tan(1, 1) = 2 * p2 * x + 6 * p1 * y;

        Jp = J_rad_mul_xy + J_tan;
    }

    return {xd, yd};
}



template <typename T>
Eigen::Matrix<T, 2, 1> undistortPointGN(const Eigen::Matrix<T, 2, 1>& point, const Eigen::Matrix<T, 2, 1>& guess,
                                        const DistortionBase<T>& d, int iterations = 5)
{
    Eigen::Matrix<T, 2, 1> x = guess;

    T last_chi2     = std::numeric_limits<T>::infinity();
    Vec2 last_point = guess;

    for (int it = 0; it < iterations; ++it)
    {
        Mat2 J;
        Vec2 res = distortNormalizedPoint(x, d, &J) - point;

        T chi2 = res.squaredNorm();

        Mat2 JtJ = J.transpose() * J;
        Vec2 Jtb = -J.transpose() * res;


        if (chi2 > last_chi2)
        {
            x = last_point;
            continue;
        }


        last_point = x;
        last_chi2  = chi2;
        Vec2 delta = JtJ.ldlt().solve(Jtb);
        x += delta;
    }


    // Final check
    T chi2 = (distortNormalizedPoint(x, d) - point).squaredNorm();
    if (chi2 > last_chi2)
    {
        x = last_point;
    }


    return x;
}



/**
 * The inverse OpenCV distortion model with 5 parameters.
 */
template <typename T>
Eigen::Matrix<T, 2, 1> undistortNormalizedPoint1235(const Eigen::Matrix<T, 2, 1>& point,
                                                    const DistortionBase<T>& distortion, int iterations = 5)
{
    T x  = point.x();
    T y  = point.y();
    T k1 = distortion.k1;
    T k2 = distortion.k2;
    T k3 = distortion.k3;
    T p1 = distortion.p1;
    T p2 = distortion.p2;

    T x0 = x;
    T y0 = y;
    // compensate distortion iteratively
    for (int j = 0; j < iterations; j++)
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
}



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
        tmp       = undistortPointGN(tmp, tmp, dis);
        tmp       = intr.normalizedToImage(tmp);
        *__output = tmp;
    }
}

}  // namespace Saiga
