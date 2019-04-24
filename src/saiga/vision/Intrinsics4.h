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

    T fx, fy;
    T cx, cy;

    Intrinsics4Base() {}
    Intrinsics4Base(T fx, T fy, T cx, T cy) : fx(fx), fy(fy), cx(cx), cy(cy) {}
    Intrinsics4Base(const Vec4& v) : fx(v(0)), fy(v(1)), cx(v(2)), cy(v(3)) {}

    Vec2 project(const Vec3& X) const
    {
        auto x = X(0) / X(2);
        auto y = X(1) / X(2);
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
    Intrinsics4Base<G> cast()
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
    StereoCamera4Base(const Vec5& v) : Intrinsics4Base<T>(v.segment<4>(0)), bf(v(4)) {}

    template <typename G>
    StereoCamera4Base<G> cast()
    {
        return {Intrinsics4Base<T>::template cast<G>(), bf};
    }

    Vec5 coeffs() const { return {this->fx, this->fy, this->cx, this->cy, bf}; }
    void coeffs(Vec5 v) { (*this) = v; }
};

using StereoCamera4  = StereoCamera4Base<double>;
using StereoCamera4f = StereoCamera4Base<float>;


using Distortion = Eigen::Matrix<double, 5, 1>;


}  // namespace Saiga
