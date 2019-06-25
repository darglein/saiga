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
struct BAPointMono
{
    static constexpr int ResCount      = 2;
    static constexpr int VarCountPoint = 3;

    using ResidualType    = Eigen::Matrix<T, ResCount, 1>;
    using PointJacobiType = Eigen::Matrix<T, ResCount, VarCountPoint>;

    using PointResidualType = Eigen::Matrix<T, VarCountPoint, 1>;  // Jpoint^T * residual

    using PointDiaBlockType = Eigen::Matrix<T, VarCountPoint, VarCountPoint>;

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

    EIGEN_ALWAYS_INLINE static void evaluateResidualAndJacobian(const CameraType& camera, const SE3Type& extr,
                                                                const Vec3& wp, const Vec2& observed, T weight,
                                                                ResidualType& res, PointJacobiType& JrowPoint)
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
        // =================== Point ================

        auto R = extr.so3().matrix();

        JrowPoint(0, 0) = -R(0, 0) * zinv + x * R(2, 0) * zzinv;
        JrowPoint(0, 1) = -R(0, 1) * zinv + x * R(2, 1) * zzinv;
        JrowPoint(0, 2) = -R(0, 2) * zinv + x * R(2, 2) * zzinv;

        JrowPoint(1, 0) = -R(1, 0) * zinv + y * R(2, 0) * zzinv;
        JrowPoint(1, 1) = -R(1, 1) * zinv + y * R(2, 1) * zzinv;
        JrowPoint(1, 2) = -R(1, 2) * zinv + y * R(2, 2) * zzinv;

        JrowPoint.row(0) *= camera.fx * weight;
        JrowPoint.row(1) *= camera.fy * weight;
    }
};



template <typename T>
struct BAPointStereo
{
    static constexpr int ResCount      = 3;
    static constexpr int VarCountPoint = 3;

    using ResidualType    = Eigen::Matrix<T, ResCount, 1>;
    using PointJacobiType = Eigen::Matrix<T, ResCount, VarCountPoint>;

    using PointResidualType = Eigen::Matrix<T, VarCountPoint, 1>;  // Jpoint^T * residual

    using PointDiaBlockType = Eigen::Matrix<T, VarCountPoint, VarCountPoint>;

    using CameraType = StereoCamera4Base<T>;
    using SE3Type    = Sophus::SE3<T>;
    using Vec3       = Eigen::Matrix<T, 3, 1>;
    using Vec2       = Eigen::Matrix<T, 2, 1>;


    EIGEN_ALWAYS_INLINE static ResidualType evaluateResidual(const CameraType& camera, const SE3Type& extr,
                                                             const Vec3& wp, const Vec2& observed, T observedDepth,
                                                             T weight)
    {
        Vec3 pc   = extr * wp;
        Vec3 proj = camera.project3(pc);

        Vec3 obs3;
        obs3 << observed(0), observed(1), observedDepth;

        //        Vec3 res = reprojectionErrorDepth(obs3, proj, camera.bf);
        T stereoPointObs = obs3(0) - camera.bf / obs3(2);
        T stereoPoint    = proj(0) - camera.bf / proj(2);
        Vec3 res         = {observed(0) - proj(0), obs3(1) - proj(1), stereoPointObs - stereoPoint};


        res *= weight;
        return res;
    }

    EIGEN_ALWAYS_INLINE static void evaluateResidualAndJacobian(const CameraType& camera, const SE3Type& extr,
                                                                const Vec3& wp, const Vec2& observed, T observedDepth,
                                                                T weight, ResidualType& res, PointJacobiType& JrowPoint)
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

        auto disparity = proj(0) - camera.bf / pc(2);

        T stereoPointObs = observed(0) - camera.bf / observedDepth;
        double diff2     = stereoPointObs - disparity;

        res(2) = diff2;

        //        T stereoPoint    = proj(0) - camera.bf / proj(2);
        //        res              = {observed(0) - proj(0), obs3(1) - proj(1), stereoPointObs - stereoPoint};
        res *= weight;


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

        JrowPoint.row(0) *= weight;
        JrowPoint.row(1) *= weight;
        JrowPoint.row(2) *= weight;
    }
};

}  // namespace Kernel
}  // namespace Saiga
