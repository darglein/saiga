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
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;

    T fx, fy;
    T cx, cy;

    Intrinsics4Base() {}
    Intrinsics4Base(T fx, T fy, T cx, T cy) : fx(fx), fy(fy), cx(cx), cy(cy) {}

    Vec2 project(const Vec3& X) const
    {
        auto x = X(0) / X(2);
        auto y = X(1) / X(2);
        return {fx * x + cx, fy * y + cy};
    }

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

    Mat3 K()
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
        return {fx, fy, cx, cy};
    }
};

using Intrinsics4  = Intrinsics4Base<double>;
using Intrinsics4f = Intrinsics4Base<float>;


}  // namespace Saiga
