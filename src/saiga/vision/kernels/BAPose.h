/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
namespace Kernel
{
template <typename T, bool AlignVec4 = false, bool normalizedImageSpace = false>
struct BAPoseMono
{
    static constexpr int ResCount     = 2;
    static constexpr int VarCountPose = AlignVec4 ? 8 : 6;

    using ResidualType = Eigen::Matrix<T, ResCount, 1>;
    //    using ResidualBlockType = Eigen::Matrix<T, VarCountPose, 1>;
    using JacobiType = Eigen::Matrix<T, ResCount, VarCountPose>;
    //    using PoseJacobiType    = Eigen::Matrix<T, ResCount, VarCountPose, Eigen::RowMajor>;
    //    using PoseDiaBlockType  = Eigen::Matrix<T, VarCountPose, VarCountPose, Eigen::RowMajor>;


    using CameraType = Intrinsics4Base<T>;
    using SE3Type    = Sophus::SE3<T>;

    using Vec2       = Eigen::Matrix<T, 2, 1>;
    using Vec3       = Eigen::Matrix<T, 3, 1>;
    using Vec4       = Eigen::Matrix<T, 4, 1>;
    using WorldPoint = Vec3;

    static inline ResidualType evaluateResidual(const CameraType& camera, const SE3Type& extr, const WorldPoint& wp,
                                                const Vec2& observed, T weight)
    {
        Vec3 pc   = extr * wp;
        Vec2 proj = camera.project(pc);
        Vec2 res  = observed - proj;
        res *= weight;
        return res;
    }

    static inline bool evaluateResidualAndJacobian(const CameraType& camera, const SE3Type& extr, const WorldPoint& wp,
                                                   const Vec2& observed, ResidualType& res, JacobiType& Jrow, T weight)
    {
        Vec3 pc = extr * wp;

        auto x     = pc(0);
        auto y     = pc(1);
        auto z     = pc(2);
        auto zz    = z * z;
        auto zinv  = 1 / z;
        auto zzinv = 1 / zz;


        Vec2 proj;
        if constexpr (normalizedImageSpace)
            proj = Vec2(x, y) * zinv;
        else
            proj = camera.project(pc);

        res = observed - proj;
        res *= weight;


        auto fxw = weight;
        auto fyw = weight;
        if constexpr (!normalizedImageSpace)
        {
            fxw *= camera.fx;
            fyw *= camera.fy;
        }

        // Tx
        Jrow(0, 0) = fxw * zinv;
        Jrow(0, 1) = 0;
        Jrow(0, 2) = fxw * -x * zzinv;
        // Rx
        Jrow(0, 3) = fyw * -y * x * zzinv;
        Jrow(0, 4) = fyw * (1 + (x * x) * zzinv);
        Jrow(0, 5) = fyw * -y * zinv;
        //        Jrow.row(0) *= camera.fx * weight;

        // Ty
        Jrow(1, 0) = 0;
        Jrow(1, 1) = fxw * zinv;
        Jrow(1, 2) = fxw * -y * zzinv;
        // Ry
        Jrow(1, 3) = fyw * (-1 - (y * y) * zzinv);
        Jrow(1, 4) = fyw * x * y * zzinv;
        Jrow(1, 5) = fyw * x * zinv;
        //        Jrow.row(1) *= camera.fy * weight;

        return z > 0;
    }
};  // namespace Kernel


template <typename T, bool AlignVec4 = false, bool normalizedImageSpace = false>
struct BAPoseStereo
{
    static constexpr int ResCount     = 3;
    static constexpr int VarCountPose = AlignVec4 ? 8 : 6;

    using ResidualType = Eigen::Matrix<T, ResCount, 1>;
    using JacobiType   = Eigen::Matrix<T, ResCount, VarCountPose>;

    using CameraType = StereoCamera4Base<T>;
    using SE3Type    = Sophus::SE3<T>;

    using Vec2       = Eigen::Matrix<T, 2, 1>;
    using Vec3       = Eigen::Matrix<T, 3, 1>;
    using Vec4       = Eigen::Matrix<T, 4, 1>;
    using WorldPoint = Vec3;

    static ResidualType evaluateResidual(const CameraType& camera, const SE3Type& extr, const WorldPoint& wp,
                                         const Vec2& observed, T observedDepth, T weight)
    {
        Vec3 pc   = extr * wp;
        Vec3 proj = camera.projectStereo(pc);

        Vec3 obs3(observed(0), observed(1), observed(0) - camera.bf / observedDepth);

        Vec3 res = obs3 - proj;
        res *= weight;
        return res;
    }

    static bool evaluateResidualAndJacobian(const CameraType& camera, const SE3Type& extr, const WorldPoint& wp,
                                            const Vec2& observed, T observedDepth, ResidualType& res, JacobiType& Jrow,
                                            T weight)
    {
        Vec3 pc   = extr * wp;
        Vec3 proj = camera.projectStereo(pc);

        Vec3 obs3(observed(0), observed(1), observed(0) - camera.bf / observedDepth);

        res = obs3 - proj;
        res *= weight;

        auto x     = pc(0);
        auto y     = pc(1);
        auto z     = pc(2);
        auto zz    = z * z;
        auto zinv  = 1 / z;
        auto zzinv = 1 / zz;

        auto fxw = camera.fx * weight;
        auto fyw = camera.fy * weight;
        auto bfw = camera.bf * weight;

        // Translation
        Jrow(0, 0) = fxw * zinv;
        Jrow(0, 1) = 0;
        Jrow(0, 2) = fxw * -x * zzinv;
        Jrow(1, 0) = 0;
        Jrow(1, 1) = fyw * zinv;
        Jrow(1, 2) = fyw * -y * zzinv;

        // Rotation
        Jrow(0, 3) = fxw * -y * x * zzinv;
        Jrow(0, 4) = fxw * (1 + (x * x) * zzinv);
        Jrow(0, 5) = fxw * -y * zinv;
        Jrow(1, 3) = fyw * (-1 - (y * y) * zzinv);
        Jrow(1, 4) = fyw * x * y * zzinv;
        Jrow(1, 5) = fyw * x * zinv;


        Jrow(2, 0) = Jrow(0, 0);
        Jrow(2, 1) = 0;
        Jrow(2, 2) = Jrow(0, 2) + bfw * zzinv;

        Jrow(2, 3) = Jrow(0, 3) + bfw * y * zzinv;
        Jrow(2, 4) = Jrow(0, 4) - bfw * x * zzinv;
        Jrow(2, 5) = Jrow(0, 5);

        return z > 0;
    }
};

}  // namespace Kernel
}  // namespace Saiga
