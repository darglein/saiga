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
template <typename T>
struct BAPosePointMono
{
    static constexpr int ResCount      = 2;
    static constexpr int VarCountPose  = 6;
    static constexpr int VarCountPoint = 3;

    using ResidualType    = Eigen::Matrix<T, ResCount, 1>;
    using PoseJacobiType  = Eigen::Matrix<T, ResCount, VarCountPose>;
    using PointJacobiType = Eigen::Matrix<T, ResCount, VarCountPoint>;

    using PoseResidualType  = Eigen::Matrix<T, VarCountPose, 1>;   // Jpose^T * residual
    using PointResidualType = Eigen::Matrix<T, VarCountPoint, 1>;  // Jpoint^T * residual

    using PoseDiaBlockType        = Eigen::Matrix<T, VarCountPose, VarCountPose>;
    using PointDiaBlockType       = Eigen::Matrix<T, VarCountPoint, VarCountPoint>;
    using PosePointUpperBlockType = Eigen::Matrix<T, VarCountPose, VarCountPoint>;  // assuming first poses then points
    using PointPoseUpperBlockType = Eigen::Matrix<T, VarCountPoint, VarCountPose>;  // first points then poses

    using CameraType = Intrinsics4Base<T>;
    using SE3Type    = Sophus::SE3<T>;
    using Vec3       = Eigen::Matrix<T, 3, 1>;
    using Vec2       = Eigen::Matrix<T, 2, 1>;

    static ResidualType evaluateResidual(const CameraType& camera, const SE3Type& extr, const Vec3& wp,
                                         const Vec2& observed, T weight)
    {
        Vec3 pc   = extr * wp;
        Vec2 proj = camera.project(pc);
        Vec2 res  = observed - proj;
        res *= weight;
        return res;
    }

    static void evaluateResidualAndJacobian(const CameraType& camera, const SE3Type& extr, const Vec3& wp,
                                            const Vec2& observed, T weight, ResidualType& res, PoseJacobiType& JrowPose,
                                            PointJacobiType& JrowPoint)
    {
        Vec3 pc = extr * wp;

        auto x     = pc(0);
        auto y     = pc(1);
        auto z     = pc(2);
        auto zz    = z * z;
        auto zinv  = 1 / z;
        auto zzinv = 1 / zz;

        // =================== Residual ================
        Vec2 proj = camera.project(pc);
        res       = observed - proj;
        res *= weight;


        // =================== Pose ================
        // Translation
        JrowPose(0, 0) = zinv;
        JrowPose(0, 1) = 0;
        JrowPose(0, 2) = -x * zzinv;
        JrowPose(1, 0) = 0;
        JrowPose(1, 1) = zinv;
        JrowPose(1, 2) = -y * zzinv;


        // Rotation
        JrowPose(0, 3) = -y * x * zzinv;
        JrowPose(0, 4) = (1 + (x * x) * zzinv);
        JrowPose(0, 5) = -y * zinv;
        JrowPose(1, 3) = (-1 - (y * y) * zzinv);
        JrowPose(1, 4) = x * y * zzinv;
        JrowPose(1, 5) = x * zinv;

        JrowPose.row(0) *= camera.fx * weight;
        JrowPose.row(1) *= camera.fy * weight;


        // =================== Point ================

        auto R = extr.so3().matrix();

        JrowPoint(0, 0) = R(0, 0) * zinv - x * R(2, 0) * zzinv;
        JrowPoint(0, 1) = R(0, 1) * zinv - x * R(2, 1) * zzinv;
        JrowPoint(0, 2) = R(0, 2) * zinv - x * R(2, 2) * zzinv;

        JrowPoint(1, 0) = R(1, 0) * zinv - y * R(2, 0) * zzinv;
        JrowPoint(1, 1) = R(1, 1) * zinv - y * R(2, 1) * zzinv;
        JrowPoint(1, 2) = R(1, 2) * zinv - y * R(2, 2) * zzinv;

        JrowPoint.row(0) *= camera.fx * weight;
        JrowPoint.row(1) *= camera.fy * weight;
    }
};



}  // namespace Kernel
}  // namespace Saiga
