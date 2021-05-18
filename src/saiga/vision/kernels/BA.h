/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
// Returns: <Residual, Depth>
inline std::pair<Vec2, double> BundleAdjustment(const Intrinsics4& camera, const Vec2& observation, const SE3& pose,
                                                const Vec3& point, double weight,
                                                Matrix<double, 2, 6>* jacobian_pose  = nullptr,
                                                Matrix<double, 2, 3>* jacobian_point = nullptr)
{
    Vec3 p      = pose * point;
    Vec2 p_by_z = Vec2(p(0) / p(2), p(1) / p(2));

    Vec2 residual;
    residual(0) = camera.fx * p_by_z(0) + camera.cx - observation(0);
    residual(1) = camera.fy * p_by_z(1) + camera.cy - observation(1);
    residual *= weight;

    auto x     = p(0);
    auto y     = p(1);
    auto z     = p(2);
    auto zz    = z * z;
    auto zinv  = 1 / z;
    auto zzinv = 1 / zz;
    if (jacobian_pose)
    {
        // divion by z (this is a little verbose and can be simplified by a lot).
        (*jacobian_pose)(0, 0) = zinv;
        (*jacobian_pose)(0, 1) = 0;
        (*jacobian_pose)(0, 2) = -x * zzinv;
        (*jacobian_pose)(1, 0) = 0;
        (*jacobian_pose)(1, 1) = zinv;
        (*jacobian_pose)(1, 2) = -y * zzinv;

        // rotation
        (*jacobian_pose)(0, 3) = -y * x * zzinv;
        (*jacobian_pose)(0, 4) = (1 + (x * x) * zzinv);
        (*jacobian_pose)(0, 5) = -y * zinv;
        (*jacobian_pose)(1, 3) = (-1 - (y * y) * zzinv);
        (*jacobian_pose)(1, 4) = x * y * zzinv;
        (*jacobian_pose)(1, 5) = x * zinv;

        (*jacobian_pose).row(0) *= camera.fx;
        (*jacobian_pose).row(1) *= camera.fy;
        (*jacobian_pose) *= weight;
    }

    if (jacobian_point)
    {
        auto R = pose.so3().matrix();

        (*jacobian_point)(0, 0) = (R(0, 0) - p_by_z(0) * R(2, 0)) * zinv;
        (*jacobian_point)(0, 1) = (R(0, 1) - p_by_z(0) * R(2, 1)) * zinv;
        (*jacobian_point)(0, 2) = (R(0, 2) - p_by_z(0) * R(2, 2)) * zinv;
        (*jacobian_point)(1, 0) = (R(1, 0) - p_by_z(1) * R(2, 0)) * zinv;
        (*jacobian_point)(1, 1) = (R(1, 1) - p_by_z(1) * R(2, 1)) * zinv;
        (*jacobian_point)(1, 2) = (R(1, 2) - p_by_z(1) * R(2, 2)) * zinv;

        (*jacobian_point).row(0) *= camera.fx;
        (*jacobian_point).row(1) *= camera.fy;

        (*jacobian_point) *= weight;
    }
    return {residual, p(2)};
}

// Returns: <Residual, Depth>
inline std::pair<Vec3, double> BundleAdjustmentStereo(const StereoCamera4& camera, const Vec2& observation,
                                                      double observed_stereo_point, const SE3& pose, const Vec3& point,
                                                      double weight, double weight_depth,
                                                      Matrix<double, 3, 6>* jacobian_pose  = nullptr,
                                                      Matrix<double, 3, 3>* jacobian_point = nullptr)
{
    Vec3 p      = pose * point;
    Vec2 p_by_z = Vec2(p(0) / p(2), p(1) / p(2));

    Vec2 projected_point(camera.fx * p_by_z(0) + camera.cx, camera.fy * p_by_z(1) + camera.cy);

    Vec3 residual;
    residual.head<2>() = projected_point - observation;
    //    residual(0) = camera.fx * p_by_z(0) + camera.cx - observation(0);
    //    residual(1) = camera.fy * p_by_z(1) + camera.cy - observation(1);


    double stereo_point = projected_point(0) - camera.bf / p(2);
    residual(2)         = observed_stereo_point - stereo_point;


    residual(0) *= weight;
    residual(1) *= weight;
    residual(2) *= weight_depth;

    auto x     = p(0);
    auto y     = p(1);
    auto z     = p(2);
    auto zz    = z * z;
    auto zinv  = 1 / z;
    auto zzinv = 1 / zz;
    if (jacobian_pose)
    {
        auto& J = *jacobian_pose;

        // divion by z (this is a little verbose and can be simplified by a lot).
        J(0, 0) = zinv;
        J(0, 1) = 0;
        J(0, 2) = -x * zzinv;
        J(1, 0) = 0;
        J(1, 1) = zinv;
        J(1, 2) = -y * zzinv;

        // rotation
        J(0, 3) = -y * x * zzinv;
        J(0, 4) = (1 + (x * x) * zzinv);
        J(0, 5) = -y * zinv;
        J(1, 3) = (-1 - (y * y) * zzinv);
        J(1, 4) = x * y * zzinv;
        J(1, 5) = x * zinv;

        J.row(0) *= camera.fx;
        J.row(1) *= camera.fy;

        // depth
        J(2, 0) = -J(0, 0);
        J(2, 1) = 0;
        J(2, 2) = -J(0, 2) - camera.bf * zzinv;
        J(2, 3) = -J(0, 3) - camera.bf * y * zzinv;
        J(2, 4) = -J(0, 4) + camera.bf * x * zzinv;
        J(2, 5) = -J(0, 5);

        J.row(0) *= weight;
        J.row(1) *= weight;


        J.row(2) *= weight_depth;
    }

    if (jacobian_point)
    {
        auto R  = pose.so3().matrix();
        auto& J = *jacobian_point;

        J(0, 0) = (R(0, 0) - p_by_z(0) * R(2, 0)) * zinv;
        J(0, 1) = (R(0, 1) - p_by_z(0) * R(2, 1)) * zinv;
        J(0, 2) = (R(0, 2) - p_by_z(0) * R(2, 2)) * zinv;
        J(1, 0) = (R(1, 0) - p_by_z(1) * R(2, 0)) * zinv;
        J(1, 1) = (R(1, 1) - p_by_z(1) * R(2, 1)) * zinv;
        J(1, 2) = (R(1, 2) - p_by_z(1) * R(2, 2)) * zinv;

        J.row(0) *= camera.fx;
        J.row(1) *= camera.fy;

        J(2, 0) = -J(0, 0) - camera.bf * R(2, 0) * zzinv;
        J(2, 1) = -J(0, 1) - camera.bf * R(2, 1) * zzinv;
        J(2, 2) = -J(0, 2) - camera.bf * R(2, 2) * zzinv;


        J.row(0) *= weight;
        J.row(1) *= weight;


        J.row(2) *= weight_depth;
    }
    return {residual, p(2)};
}
}  // namespace Saiga
