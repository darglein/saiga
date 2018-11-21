/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/kernels/Robust.h"

namespace Saiga
{
namespace Kernel
{
template <typename T>
struct BAPoseMono
{
    static constexpr int ResCount = 2;
    static constexpr int VarCount = 6;

    using ResidualType = Eigen::Matrix<T, ResCount, 1>;
    using JacobiType   = Eigen::Matrix<T, ResCount, VarCount>;

    using CameraType = Intrinsics4Base<T>;
    using SE3Type    = Sophus::SE3<T>;
    using Vec3       = Eigen::Matrix<T, 3, 1>;
    using Vec2       = Eigen::Matrix<T, 2, 1>;


    static void evaluateResidualAndJacobian(const CameraType& camera, const SE3Type& extr, const Vec3& wp,
                                            const Vec2& observed, ResidualType& res, JacobiType& Jrow, T weight)
    {
        Vec3 pc   = extr * wp;
        Vec2 proj = camera.project(pc);
        res       = observed - proj;

        auto x     = pc(0);
        auto y     = pc(1);
        auto z     = pc(2);
        auto zz    = z * z;
        auto zinv  = 1 / z;
        auto zzinv = 1 / zz;

        // Translation
        Jrow(0, 0) = zinv;
        Jrow(0, 1) = 0;
        Jrow(0, 2) = -x * zzinv;
        Jrow(1, 0) = 0;
        Jrow(1, 1) = zinv;
        Jrow(1, 2) = -y * zzinv;


        // Rotation
        Jrow(0, 3) = -y * x * zzinv;
        Jrow(0, 4) = (1 + (x * x) * zzinv);
        Jrow(0, 5) = -y * zinv;
        Jrow(1, 3) = (-1 - (y * y) * zzinv);
        Jrow(1, 4) = x * y * zzinv;
        Jrow(1, 5) = x * zinv;

        Jrow.row(0) *= camera.fx;
        Jrow.row(1) *= camera.fy;


        // use weight
        Jrow *= weight;
        res *= weight;
    }
};


template <typename T>
struct BAPoseStereo
{
    static constexpr int ResCount = 3;
    static constexpr int VarCount = 6;

    using ResidualType = Eigen::Matrix<T, ResCount, 1>;
    using JacobiType   = Eigen::Matrix<T, ResCount, VarCount>;

    using CameraType = Intrinsics4Base<T>;
    using SE3Type    = Sophus::SE3<T>;
    using Vec3       = Eigen::Matrix<T, 3, 1>;
    using Vec2       = Eigen::Matrix<T, 2, 1>;



    static void evaluateResidualAndJacobian(const CameraType& camera, const SE3Type& extr, const Vec3& wp,
                                            const Vec2& observed, T observedDepth, ResidualType& res, JacobiType& Jrow,
                                            T weight, T bf)
    {
        Vec3 pc   = extr * wp;
        Vec3 proj = camera.project3(pc);

        Vec3 obs3;
        obs3 << observed(0), observed(1), observedDepth;

        res = reprojectionErrorDepth(obs3, proj, bf);

        auto x     = pc(0);
        auto y     = pc(1);
        auto z     = pc(2);
        auto zz    = z * z;
        auto zinv  = 1 / z;
        auto zzinv = 1 / zz;

        // Translation
        Jrow(0, 0) = zinv;
        Jrow(0, 1) = 0;
        Jrow(0, 2) = -x * zzinv;
        Jrow(1, 0) = 0;
        Jrow(1, 1) = zinv;
        Jrow(1, 2) = -y * zzinv;


        // Rotation
        Jrow(0, 3) = -y * x * zzinv;
        Jrow(0, 4) = (1 + (x * x) * zzinv);
        Jrow(0, 5) = -y * zinv;
        Jrow(1, 3) = (-1 - (y * y) * zzinv);
        Jrow(1, 4) = x * y * zzinv;
        Jrow(1, 5) = x * zinv;


        Jrow.row(0) *= camera.fx;
        Jrow.row(1) *= camera.fy;


        Jrow(2, 0) = Jrow(0, 0);
        Jrow(2, 1) = 0;
        Jrow(2, 2) = Jrow(0, 2) + bf * zzinv;

        Jrow(2, 3) = Jrow(0, 3) + bf * y * zzinv;
        Jrow(2, 4) = Jrow(0, 4) - bf * x * zzinv;
        Jrow(2, 5) = Jrow(0, 5);



        // use weight
        Jrow *= weight;
        res *= weight;
    }
};

}  // namespace Kernel
}  // namespace Saiga
