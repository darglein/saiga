/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"


namespace Saiga
{
template <typename T>
struct Intrinsics4Base
{
    using Vec4 = Eigen::Matrix<T, 4, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;

    // Initialized to the identity "matrix"
    T fx = 1, fy = 1;
    T cx = 0, cy = 0;

    Intrinsics4Base() {}
    Intrinsics4Base(T fx, T fy, T cx, T cy) : fx(fx), fy(fy), cx(cx), cy(cy) {}
    Intrinsics4Base(const Vec4& v) : fx(v(0)), fy(v(1)), cx(v(2)), cy(v(3)) {}
    Intrinsics4Base(const Mat3& K) : fx(K(0, 0)), fy(K(1, 1)), cx(K(0, 2)), cy(K(1, 2)) {}

    Intrinsics4Base<T> inverse() const { return {T(1) / fx, T(1) / fy, -cx / fx, -cy / fy}; }

    Vec2 project(const Vec3& X) const
    {
        auto invz = T(1) / X(2);
        auto x    = X(0) * invz;
        auto y    = X(1) * invz;
        return {fx * x + cx, fy * y + cy};
    }

    Vec2 operator*(const Vec3& X) const { return project(X); }

    // same as above, but keep the z value in the output
    Vec3 project3(const Vec3& X) const
    {
        Vec2 p = project(X);
        return {p(0), p(1), X(2)};
    }

    Vec3 unproject(const Vec2& ip, T depth) const
    {
        Vec3 p((ip(0) - cx) / fx, (ip(1) - cy) / fy, 1);
        return p * depth;
    }

    // from image to normalized image space
    // same as above but without depth
    Vec2 unproject2(const Vec2& ip) const { return {(ip(0) - cx) / fx, (ip(1) - cy) / fy}; }

    Vec2 normalizedToImage(const Vec2& p) const { return {fx * p(0) + cx, fy * p(1) + cy}; }

    void scale(T s)
    {
        fx *= s;
        fy *= s;
        cx *= s;
        cy *= s;
    }

    Mat3 matrix()
    {
        Mat3 k;
        // clang-format off
        k <<
            fx, 0,  cx,
            0,  fy, cy,
            0,  0,  1;
        // clang-format on
        return k;
    }

    template <typename G>
    Intrinsics4Base<G> cast() const
    {
        return {(G)fx, (G)fy, (G)cx, (G)cy};
    }

    // convert to eigen vector
    Vec4 coeffs() const { return {fx, fy, cx, cy}; }
    void coeffs(Vec4 v) { (*this) = v; }
};

using Intrinsics4  = Intrinsics4Base<double>;
using Intrinsics4f = Intrinsics4Base<float>;

template <typename T>
struct StereoCamera4Base : public Intrinsics4Base<T>
{
    using Vec5 = Eigen::Matrix<T, 5, 1>;
    using Base = Intrinsics4Base<T>;
    using Base::cx;
    using Base::cy;
    using Base::fx;
    using Base::fy;
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
    StereoCamera4Base(T fx, T fy, T cx, T cy, T bf) : Intrinsics4Base<T>(fx, fy, cx, cy), bf(bf) {}
    StereoCamera4Base(const Intrinsics4Base<T>& i, T bf) : Intrinsics4Base<T>(i), bf(bf) {}
    StereoCamera4Base(const Vec5& v) : Intrinsics4Base<T>(Vec4(v.template segment<4>(0))), bf(v(4)) {}

    StereoCamera4Base<T> inverse() const { return {T(1) / fx, T(1) / fy, -cx / fx, -cy / fy, T(1) / bf}; }

    template <typename G>
    StereoCamera4Base<G> cast()
    {
        return {Intrinsics4Base<T>::template cast<G>(), static_cast<G>(bf)};
    }

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

    Vec5 coeffs() const
    {
        Vec5 v;
        v << this->fx, this->fy, this->cx, this->cy, bf;
        return v;
    }
    void coeffs(Vec5 v) { (*this) = v; }

    // Baseline in meters
    T baseLine() { return bf / fx; }
};

using StereoCamera4  = StereoCamera4Base<double>;
using StereoCamera4f = StereoCamera4Base<float>;



}  // namespace Saiga
