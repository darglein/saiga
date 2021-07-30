/**
 * Copyright (c) 2021 Darius Rückert
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

    HD inline DistortionBase() {}

    HD inline DistortionBase(const Eigen::Matrix<T, 8, 1>& c)
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

    HD inline Eigen::Matrix<T, 8, 1> Coeffs() const
    {
        Eigen::Matrix<T, 8, 1> result;
        result << k1, k2, k3, k4, k5, k6, p1, p2;
        return result;
    }


    HD inline Eigen::Matrix<T, 8, 1> OpenCVOrder() const
    {
        Eigen::Matrix<T, 8, 1> result;
        result << k1, k2, p1, p2, k3, k4, k5, k6;
        return result;
    }

    template <typename G>
    HD inline DistortionBase<G> cast() const
    {
        return Coeffs().template cast<G>().eval();
    }

    HD inline T RadialFactor(const Eigen::Matrix<T, 2, 1>& p)
    {
        T r2       = p.dot(p);
        T r4       = r2 * r2;
        T r6       = r2 * r4;
        T radial_u = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
        T radial_v = T(1) + k4 * r2 + k5 * r4 + k6 * r6;

        T radial_iv = T(1) / radial_v;
        T radial    = radial_u * radial_iv;
        return radial;
    }

    // Due to the higher oder polynomial the radial distortion is not monotonic.
    // This results in a non-bijective mapping, where multiple world points are mapped to the same image points.
    // To prevent this case, this function computes a threshold which should be passed as 'max_r' to
    // distortNormalizedPoint.
    HD inline T MonotonicThreshold(int steps = 100, T max_r = 2)
    {
        float last = 0;
        float th   = max_r;
        for (int i = 0; i < steps; ++i)
        {
            float a     = (float(i) / steps) * max_r;
            float c     = RadialFactor(vec2(a, 0));
            float shift = c * a;
            if (shift - last < 0)
            {
                th = a;
                break;
            }
            last = shift;
        }
        return th;
    }
};

// Use double by default here
using Distortion  = DistortionBase<double>;
using Distortionf = DistortionBase<float>;



template <typename T>
std::ostream& operator<<(std::ostream& strm, const DistortionBase<T> dist)
{
    strm << dist.Coeffs().transpose();
    return strm;
}


/**
 * The OpenCV distortion model applied to a point in normalized image coordinates.
 * You can find a glsl implementation in shader/vision/distortion.h
 */
template <typename T>
HD inline Eigen::Matrix<T, 2, 1> distortNormalizedPoint(const Eigen::Matrix<T, 2, 1>& point,
                                                        const DistortionBase<T>& distortion,
                                                        Matrix<T, 2, 2>* J_point      = nullptr,
                                                        Matrix<T, 2, 8>* J_distortion = nullptr, T max_r = T(100000))
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
    T r4 = r2 * r2;
    T r6 = r4 * r2;

    T radial_u = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
    T radial_v = T(1) + k4 * r2 + k5 * r4 + k6 * r6;

    T radial_iv = T(1) / radial_v;
    T radial    = radial_u * radial_iv;


    T tangentialX = p1 * _2xy + p2 * (r2 + T(2) * x2);
    T tangentialY = p1 * (r2 + T(2) * y2) + p2 * _2xy;

    T xd = x * radial + tangentialX;
    T yd = y * radial + tangentialY;

    if (r2 > max_r * max_r)
    {
        xd = 100000;
        yd = 100000;
    }

    if (J_point)
    {
        Matrix<T, 2, 2> J_rad_u;
        J_rad_u(0, 0) = k1 * (2 * x);
        J_rad_u(0, 1) = k1 * (2 * y);
        J_rad_u(0, 0) += k2 * (4 * x * x * x + 4 * x * y * y);
        J_rad_u(0, 1) += k2 * (4 * y * y * y + 4 * y * x * x);
        J_rad_u(0, 0) += k3 * (6 * x2 * x2 * x + 12 * x2 * x * y2 + 6 * x * y2 * y2);
        J_rad_u(0, 1) += k3 * (6 * y2 * y2 * y + 12 * y2 * y * x2 + 6 * y * x2 * x2);
        J_rad_u(1, 0) = J_rad_u(0, 0);
        J_rad_u(1, 1) = J_rad_u(0, 1);


        Matrix<T, 2, 2> J_rad_v;
        J_rad_v(0, 0) = k4 * (2 * x);
        J_rad_v(0, 1) = k4 * (2 * y);
        J_rad_v(0, 0) += k5 * (4 * x * x * x + 4 * x * y * y);
        J_rad_v(0, 1) += k5 * (4 * y * y * y + 4 * y * x * x);
        J_rad_v(0, 0) += k6 * (6 * x2 * x2 * x + 12 * x2 * x * y2 + 6 * x * y2 * y2);
        J_rad_v(0, 1) += k6 * (6 * y2 * y2 * y + 12 * y2 * y * x2 + 6 * y * x2 * x2);
        J_rad_v(1, 0) = J_rad_v(0, 0);
        J_rad_v(1, 1) = J_rad_v(0, 1);

        Matrix<T, 2, 2> J_rad;
        J_rad = (J_rad_u * radial_v - J_rad_v * radial_u) * radial_iv * radial_iv;

        Matrix<T, 2, 2> J_rad_mul_xy;
        J_rad_mul_xy(0, 0) = x * J_rad(0, 0) + radial;
        J_rad_mul_xy(0, 1) = x * J_rad(0, 1);
        J_rad_mul_xy(1, 0) = y * J_rad(1, 0);
        J_rad_mul_xy(1, 1) = y * J_rad(1, 1) + radial;

        Matrix<T, 2, 2> J_tan;
        J_tan(0, 0) = 2 * p1 * y + 6 * p2 * x;
        J_tan(0, 1) = 2 * p1 * x + 2 * p2 * y;

        J_tan(1, 0) = 2 * p2 * y + 2 * p1 * x;
        J_tan(1, 1) = 2 * p2 * x + 6 * p1 * y;

        *J_point = J_rad_mul_xy + J_tan;
    }

    if (J_distortion)
    {
        auto& J = *J_distortion;
        J(0, 0) = r2 * x * radial_iv;
        J(0, 1) = r4 * x * radial_iv;
        J(0, 2) = r6 * x * radial_iv;
        J(1, 0) = r2 * y * radial_iv;
        J(1, 1) = r4 * y * radial_iv;
        J(1, 2) = r6 * y * radial_iv;

        J(0, 3) = -r2 * radial_u * radial_iv * radial_iv * x;
        J(0, 4) = -r4 * radial_u * radial_iv * radial_iv * x;
        J(0, 5) = -r6 * radial_u * radial_iv * radial_iv * x;
        J(1, 3) = -r2 * radial_u * radial_iv * radial_iv * y;
        J(1, 4) = -r4 * radial_u * radial_iv * radial_iv * y;
        J(1, 5) = -r6 * radial_u * radial_iv * radial_iv * y;

        J(0, 6) = _2xy;
        J(1, 7) = _2xy;
        J(0, 7) = (r2 + T(2) * x2);
        J(1, 6) = (r2 + T(2) * y2);
    }

    return {xd, yd};
}


template <typename T>
Eigen::Matrix<T, 2, 1> undistortPointGN(const Eigen::Matrix<T, 2, 1>& point, const Eigen::Matrix<T, 2, 1>& guess,
                                        const DistortionBase<T>& d, int iterations = 5)
{
    Eigen::Matrix<T, 2, 1> x          = guess;
    Eigen::Matrix<T, 2, 1> last_point = guess;

    T last_chi2 = std::numeric_limits<T>::infinity();

    for (int it = 0; it < iterations; ++it)
    {
        Eigen::Matrix<T, 2, 2> J;
        Eigen::Matrix<T, 2, 1> res = distortNormalizedPoint(x, d, &J) - point;

        T chi2 = res.squaredNorm();

        Eigen::Matrix<T, 2, 2> JtJ = J.transpose() * J;
        Eigen::Matrix<T, 2, 1> Jtb = -J.transpose() * res;


        if (chi2 > last_chi2)
        {
            x = last_point;
            continue;
        }


        last_point = x;
        last_chi2  = chi2;

        Eigen::Matrix<T, 2, 1> delta = JtJ.ldlt().solve(Jtb);
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
HD inline Eigen::Matrix<T, 2, 1> undistortNormalizedPointSimple(const Eigen::Matrix<T, 2, 1>& point,
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
                         const IntrinsicsPinhole<_T>& intr, const DistortionBase<_T>& dis)
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
