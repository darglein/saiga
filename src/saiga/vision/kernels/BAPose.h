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
template <typename T, bool AlignVec4 = false>
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
    using WorldPoint = typename std::conditional<AlignVec4, Vec4, Vec3>::type;

    static inline ResidualType evaluateResidual(const CameraType& camera, const SE3Type& extr, const WorldPoint& wp,
                                                const Vec2& observed, T weight)
    {
        Vec3 pc   = extr * wp.template segment<3>(0);
        Vec2 proj = camera.project(pc);
        Vec2 res  = observed - proj;
        res *= weight;
        return res;
    }

    static inline void evaluateResidualAndJacobian(const CameraType& camera, const SE3Type& extr, const WorldPoint& wp,
                                                   const Vec2& observed, ResidualType& res, JacobiType& Jrow, T weight)
    {
        Vec3 pc   = extr * wp.template segment<3>(0);
        Vec2 proj = camera.project(pc);
        res       = observed - proj;
        res *= weight;

        auto x     = pc(0);
        auto y     = pc(1);
        auto z     = pc(2);
        auto zz    = z * z;
        auto zinv  = 1 / z;
        auto zzinv = 1 / zz;

        // Tx
        Jrow(0, 0) = zinv;
        Jrow(0, 1) = 0;
        Jrow(0, 2) = -x * zzinv;
        // Rx
        Jrow(0, 3) = -y * x * zzinv;
        Jrow(0, 4) = (1 + (x * x) * zzinv);
        Jrow(0, 5) = -y * zinv;
        Jrow.row(0) *= camera.fx * weight;

        // Ty
        Jrow(1, 0) = 0;
        Jrow(1, 1) = zinv;
        Jrow(1, 2) = -y * zzinv;
        // Ry
        Jrow(1, 3) = (-1 - (y * y) * zzinv);
        Jrow(1, 4) = x * y * zzinv;
        Jrow(1, 5) = x * zinv;
        Jrow.row(1) *= camera.fy * weight;
    }
};


template <typename T, bool AlignVec4 = false>
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
    using WorldPoint = typename std::conditional<AlignVec4, Vec4, Vec3>::type;

    static ResidualType evaluateResidual(const CameraType& camera, const SE3Type& extr, const WorldPoint& wp,
                                         const Vec2& observed, T observedDepth, T weight)
    {
        Vec3 pc   = extr * wp.template segment<3>(0);
        Vec3 proj = camera.project3(pc);

        Vec3 obs3{observed(0), observed(1), observedDepth};


        T stereoPointObs = obs3(0) - camera.bf / obs3(2);
        T stereoPoint    = proj(0) - camera.bf / proj(2);
        Vec3 res         = {observed(0) - proj(0), obs3(1) - proj(1), stereoPointObs - stereoPoint};


        res *= weight;
        return res;
    }

    static void evaluateResidualAndJacobian(const CameraType& camera, const SE3Type& extr, const WorldPoint& wp,
                                            const Vec2& observed, T observedDepth, ResidualType& res, JacobiType& Jrow,
                                            T weight)
    {
        Vec3 pc   = extr * wp.template segment<3>(0);
        Vec3 proj = camera.project3(pc);

        Vec3 obs3(observed(0), observed(1), observedDepth);
        //        obs3 << observed(0), observed(1), observedDepth;


        //        res = reprojectionErrorDepth(obs3, proj, camera.bf);
        T stereoPointObs = obs3(0) - camera.bf / obs3(2);
        T stereoPoint    = proj(0) - camera.bf / proj(2);
        res = ResidualType{observed(0) - proj(0), obs3(1) - proj(1), stereoPointObs - stereoPoint} * weight;

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
        Jrow(0, 3) = camera.fx * weight * -y * x * zzinv;
        Jrow(0, 4) = camera.fx * weight * (1 + (x * x) * zzinv);
        Jrow(0, 5) = camera.fx * weight * -y * zinv;
        Jrow(1, 3) = camera.fy * weight * (-1 - (y * y) * zzinv);
        Jrow(1, 4) = camera.fy * weight * x * y * zzinv;
        Jrow(1, 5) = camera.fy * weight * x * zinv;


        //        Jrow.row(0) *= camera.fx;
        //        Jrow.row(1) *= camera.fy;


        Jrow(2, 0) = Jrow(0, 0);
        Jrow(2, 1) = 0;
        Jrow(2, 2) = Jrow(0, 2) + weight * camera.bf * zzinv;

        Jrow(2, 3) = Jrow(0, 3) + weight * camera.bf * y * zzinv;
        Jrow(2, 4) = Jrow(0, 4) - weight * camera.bf * x * zzinv;
        Jrow(2, 5) = Jrow(0, 5);



        // use weight
        //        Jrow *= weight;
        //        res *= weight;
    }
};

}  // namespace Kernel
}  // namespace Saiga
