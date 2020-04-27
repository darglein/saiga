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

    EIGEN_ALWAYS_INLINE static ResidualType evaluateResidual(const CameraType& camera, const SE3Type& extr,
                                                             const Vec3& wp, const Vec2& observed, T weight)
    {
        Vec3 pc   = extr * wp;
        Vec2 proj = camera.project(pc);
        Vec2 res  = observed - proj;
        res *= weight;
        return res;
    }

    EIGEN_ALWAYS_INLINE static bool evaluateResidualAndJacobian(const CameraType& camera, const SE3Type& extr,
                                                                const Vec3& wp, const Vec2& observed, T weight,
                                                                ResidualType& res, PoseJacobiType& JrowPose,
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
        auto x_over_z = x / z;
        auto y_over_z = y / z;

        Vec2 proj = camera.normalizedToImage({x_over_z, y_over_z});

        res = observed - proj;
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

        auto mul_x      = zinv * camera.fx * weight;
        JrowPoint(0, 0) = (R(0, 0) - x_over_z * R(2, 0)) * mul_x;
        JrowPoint(0, 1) = (R(0, 1) - x_over_z * R(2, 1)) * mul_x;
        JrowPoint(0, 2) = (R(0, 2) - x_over_z * R(2, 2)) * mul_x;

        auto mul_y      = zinv * camera.fy * weight;
        JrowPoint(1, 0) = (R(1, 0) - y_over_z * R(2, 0)) * mul_y;
        JrowPoint(1, 1) = (R(1, 1) - y_over_z * R(2, 1)) * mul_y;
        JrowPoint(1, 2) = (R(1, 2) - y_over_z * R(2, 2)) * mul_y;

        return z > 0;
    }
};



template <typename T>
struct BAPosePointStereo
{
    static constexpr int ResCount      = 3;
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

    using CameraType = StereoCamera4Base<T>;
    using SE3Type    = Sophus::SE3<T>;
    using Vec3       = Eigen::Matrix<T, 3, 1>;
    using Vec2       = Eigen::Matrix<T, 2, 1>;


    EIGEN_ALWAYS_INLINE static ResidualType evaluateResidual(const CameraType& camera, const SE3Type& extr,
                                                             const Vec3& wp, const Vec2& observed,
                                                             T observed_stereo_point, T weight, T weight_depth)
    {
        Vec3 pc   = extr * wp;
        Vec3 proj = camera.project3(pc);

        Vec3 obs3;
        obs3 << observed(0), observed(1), observed_stereo_point;

        //        Vec3 res = reprojectionErrorDepth(obs3, proj, camera.bf);
        T stereoPoint = proj(0) - camera.bf / proj(2);
        Vec3 res      = {observed(0) - proj(0), obs3(1) - proj(1), observed_stereo_point - stereoPoint};


        res(0) *= weight;
        res(1) *= weight;
        res(2) *= weight_depth;
        return res;
    }

    EIGEN_ALWAYS_INLINE static bool evaluateResidualAndJacobian(const CameraType& camera, const SE3Type& extr,
                                                                const Vec3& wp, const Vec2& observed,
                                                                T observed_stereo_point, T weight, T weight_depth,
                                                                ResidualType& res, PoseJacobiType& JrowPose,
                                                                PointJacobiType& JrowPoint)
    {
        Vec3 pc   = extr * wp;
        Vec2 proj = camera.project(pc);

        Vec2 rp = observed - proj;
        res(0)  = rp(0);
        res(1)  = rp(1);

        //        Vec3 obs3;
        //        obs3 << observed(0), observed(1), observedDepth;

        auto x     = pc(0);
        auto y     = pc(1);
        auto z     = pc(2);
        auto zz    = z * z;
        auto zinv  = 1 / z;
        auto zzinv = 1 / zz;


        // =================== Residual ================

        auto right_point = proj(0) - camera.bf / pc(2);
        res(2)           = observed_stereo_point - right_point;

        res(0) *= weight;
        res(1) *= weight;
        res(2) *= weight_depth;

        //        T stereoPoint    = proj(0) - camera.bf / proj(2);
        //        res              = {observed(0) - proj(0), obs3(1) - proj(1), stereoPointObs - stereoPoint};


        // =================== Pose ================
        // Translation
        JrowPose(0, 0) = -zinv;
        JrowPose(0, 1) = 0;
        JrowPose(0, 2) = x * zzinv;
        JrowPose(1, 0) = 0;
        JrowPose(1, 1) = -zinv;
        JrowPose(1, 2) = y * zzinv;


        // Rotation
        JrowPose(0, 3) = y * x * zzinv;
        JrowPose(0, 4) = -(1 + (x * x) * zzinv);
        JrowPose(0, 5) = y * zinv;
        JrowPose(1, 3) = -(-1 - (y * y) * zzinv);
        JrowPose(1, 4) = -x * y * zzinv;
        JrowPose(1, 5) = -x * zinv;


        JrowPose.row(0) *= camera.fx;
        JrowPose.row(1) *= camera.fy;

        JrowPose(2, 0) = JrowPose(0, 0);
        JrowPose(2, 1) = 0;
        JrowPose(2, 2) = JrowPose(0, 2) - camera.bf * zzinv;
        JrowPose(2, 3) = JrowPose(0, 3) - camera.bf * y * zzinv;
        JrowPose(2, 4) = JrowPose(0, 4) + camera.bf * x * zzinv;
        JrowPose(2, 5) = JrowPose(0, 5);

        JrowPose.row(0) *= -weight;
        JrowPose.row(1) *= -weight;
        JrowPose.row(2) *= -weight_depth;


        // =================== Point ================

        auto R = extr.so3().matrix();

        JrowPoint(0, 0) = -R(0, 0) * zinv + x * R(2, 0) * zzinv;
        JrowPoint(0, 1) = -R(0, 1) * zinv + x * R(2, 1) * zzinv;
        JrowPoint(0, 2) = -R(0, 2) * zinv + x * R(2, 2) * zzinv;

        JrowPoint(1, 0) = -R(1, 0) * zinv + y * R(2, 0) * zzinv;
        JrowPoint(1, 1) = -R(1, 1) * zinv + y * R(2, 1) * zzinv;
        JrowPoint(1, 2) = -R(1, 2) * zinv + y * R(2, 2) * zzinv;

        JrowPoint.row(0) *= camera.fx;
        JrowPoint.row(1) *= camera.fy;

        JrowPoint(2, 0) = JrowPoint(0, 0) - camera.bf * R(2, 0) * zzinv;
        JrowPoint(2, 1) = JrowPoint(0, 1) - camera.bf * R(2, 1) * zzinv;
        JrowPoint(2, 2) = JrowPoint(0, 2) - camera.bf * R(2, 2) * zzinv;

        JrowPoint.row(0) *= -weight;
        JrowPoint.row(1) *= -weight;
        JrowPoint.row(2) *= -weight_depth;

        return z > 0;
    }
};

}  // namespace Kernel
}  // namespace Saiga
