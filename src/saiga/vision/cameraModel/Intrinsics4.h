/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"


namespace Saiga
{
// Basic Intrinsics Matrix K used in the pinhole camera model.
//
// |  fx   s   cx  |
// |   0   fy  cy  |
// |   0   0    1  |
//
// The inverse is:
// |   1/fx   -s/(fx*fy)  (s*cy-cx*fy)/(fx*fy)  |
// |   0      1/fy     -cy/fy                |
// |   0      0        1                     |
template <typename T>
struct IntrinsicsPinhole
{
    using Vec5 = Eigen::Matrix<T, 5, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;
    using Mat2 = Eigen::Matrix<T, 2, 2>;

    // Initialized to identity
    T fx = 1, fy = 1;
    T cx = 0, cy = 0;
    T s = 0;

    HD inline IntrinsicsPinhole() {}
    HD inline IntrinsicsPinhole(T fx, T fy, T cx, T cy, T s) : fx(fx), fy(fy), cx(cx), cy(cy), s(s) {}
    HD inline IntrinsicsPinhole(const Vec5& v) : fx(v(0)), fy(v(1)), cx(v(2)), cy(v(3)), s(v(4)) {}
    HD inline IntrinsicsPinhole(const Mat3& K) : fx(K(0, 0)), fy(K(1, 1)), cx(K(0, 2)), cy(K(1, 2)), s(K(0, 1)) {}

    HD inline IntrinsicsPinhole<T> inverse() const
    {
        T fx2 = 1 / fx;
        T fy2 = 1 / fy;
        T cx2 = (s * cy - cx * fy) / (fx * fy);
        T cy2 = -cy / fy;
        T s2  = -s / (fx * fy);
        return {fx2, fy2, cx2, cy2, s2};
    }

    HD inline Vec2 project(const Vec3& X) const
    {
        auto invz = T(1) / X(2);
        auto x    = X(0) * invz;
        auto y    = X(1) * invz;
        return {fx * x + s * y + cx, fy * y + cy};
    }

    HD inline Vec2 operator*(const Vec3& X) const { return project(X); }

    // same as above, but keep the z value in the output
    HD inline Vec3 project3(const Vec3& X) const
    {
        Vec2 p = project(X);
        return {p(0), p(1), X(2)};
    }

    HD inline Vec3 unproject(const Vec2& ip, T depth) const
    {
        Vec3 p((ip(0) - cx) / fx, (ip(1) - cy) / fy, 1);
        return p * depth;
    }

    // from image to normalized image space
    // same as above but without depth
    HD inline Vec2 unproject2(const Vec2& ip) const { return {(ip(0) - cx) / fx, (ip(1) - cy) / fy}; }

    HD inline Vec2 normalizedToImage(const Vec2& p) const { return {fx * p(0) + s * p(1) + cx, fy * p(1) + cy}; }


    [[nodiscard]] HD inline IntrinsicsPinhole<T> scale(T s) const { return IntrinsicsPinhole<T>(Vec5(coeffs() * s)); }


    HD inline Vec2 normalizedToImage(const Vec2& p, Mat2* J_point, Matrix<T, 2, 5>* J_K) const
    {
        const Vec2 image_point = normalizedToImage(p);

        if (J_point)
        {
            (*J_point)(0, 0) = fx;
            (*J_point)(0, 1) = s;
            (*J_point)(1, 0) = 0;
            (*J_point)(1, 1) = fy;
        }

        if (J_K)
        {
            (*J_K)(0, 0) = p(0);
            (*J_K)(0, 1) = 0;
            (*J_K)(0, 2) = 1;
            (*J_K)(0, 3) = 0;
            (*J_K)(0, 4) = p(1);

            (*J_K)(1, 0) = 0;
            (*J_K)(1, 1) = p(1);
            (*J_K)(1, 2) = 0;
            (*J_K)(1, 3) = 1;
            (*J_K)(1, 4) = 0;
        }

        return image_point;
    }


    HD inline Mat3 matrix() const
    {
        Mat3 k;
        // clang-format off
        k <<
          fx, s,  cx,
            0,  fy, cy,
            0,  0,  1;
        // clang-format on
        return k;
    }

    template <typename G>
    HD inline IntrinsicsPinhole<G> cast() const
    {
        return {(G)fx, (G)fy, (G)cx, (G)cy, (G)s};
    }

    // convert to eigen vector
    HD inline Vec5 coeffs() const { return {fx, fy, cx, cy, s}; }
    HD inline void coeffs(const Vec5& v) { (*this) = v; }
};

template <typename T>
HD inline IntrinsicsPinhole<T> operator*(const IntrinsicsPinhole<T>& l, const IntrinsicsPinhole<T>& r)
{
    T fx2 = l.fx * r.fx;
    T fy2 = l.fy * r.fy;
    T s2  = l.fx * r.s + l.s * r.fy;
    T cx2 = l.fx * r.cx + l.s * r.fy + l.cx;
    T cy2 = l.fy * r.cy + l.cy;
    return {fx2, fy2, cx2, cy2, s2};
}

using IntrinsicsPinholed = IntrinsicsPinhole<double>;
using IntrinsicsPinholef = IntrinsicsPinhole<float>;

template <typename T>
std::ostream& operator<<(std::ostream& strm, const IntrinsicsPinhole<T> intr)
{
    strm << intr.coeffs().transpose();
    return strm;
}


template <typename T>
struct StereoCamera4Base : public IntrinsicsPinhole<T>
{
    using Vec6 = Eigen::Matrix<T, 6, 1>;
    using Base = IntrinsicsPinhole<T>;
    using Base::cx;
    using Base::cy;
    using Base::fx;
    using Base::fy;
    using Base::s;
    using Vec3 = typename Base::Vec3;
    /**
     * Baseline * FocalLength_x
     *
     * Used to convert from depth to disparity:
     * disparity = bf / depth
     * depth = bf / disparity
     */
    T bf;
    StereoCamera4Base() {}
    StereoCamera4Base(T fx, T fy, T cx, T cy, T s, T bf) : IntrinsicsPinhole<T>(fx, fy, cx, cy, s), bf(bf) {}
    StereoCamera4Base(const IntrinsicsPinhole<T>& i, T bf) : IntrinsicsPinhole<T>(i), bf(bf) {}
    StereoCamera4Base(const Vec6& v) : IntrinsicsPinhole<T>(Vec5(v.template head<5>())), bf(v(5)) {}

    StereoCamera4Base<T> inverse() const
    {
        T fx2 = 1 / fx;
        T fy2 = 1 / fy;
        T cx2 = (s * cy - cx * fy) / (fx * fy);
        T cy2 = -cy / fy;
        T s2  = -s / (fx * fy);
        T bf2 = 1 / bf;
        return {fx2, fy2, cx2, cy2, s2, bf2};
    }

    template <typename G>
    StereoCamera4Base<G> cast()
    {
        return {IntrinsicsPinhole<T>::template cast<G>(), static_cast<G>(bf)};
    }

    // convert an image point + its depth to the predicted point in the right image.
    // We subtract the disparity
    double LeftPointToRight(double x, double z) const { return x - bf / z; }

    // depth     = baseline * fx / disparity
    // disparity = baseline * fx / depth
    double DepthToDisparity(double depth) { return bf / depth; }
    double DisparityToDepth(double disparity) { return bf / disparity; }

    Vec3 projectStereo(const Vec3& X) const
    {
        auto invz        = T(1) / X(2);
        auto x           = X(0) * invz;
        auto y           = X(1) * invz;
        x                = fx * x + cx;
        y                = fy * y + cy;
        auto stereoPoint = x - bf * invz;
        return {x, y, stereoPoint};
    }

    Vec6 coeffs() const
    {
        Vec6 v;
        v << this->fx, this->fy, this->cx, this->cy, this->s, bf;
        return v;
    }
    void coeffs(const Vec6& v) { (*this) = v; }

    // Baseline in meters
    T baseLine() { return bf / fx; }
};

using StereoCamera4  = StereoCamera4Base<double>;
using StereoCamera4f = StereoCamera4Base<float>;

template <typename T>
std::ostream& operator<<(std::ostream& strm, const StereoCamera4Base<T> intr)
{
    strm << intr.coeffs().transpose();
    return strm;
}



}  // namespace Saiga
