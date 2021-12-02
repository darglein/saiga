/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

namespace Saiga
{


// Rotation of a point by a quaternion. The derivative of the rotation will be given in
// the lie-algebra tangent space.
//
// To update the rotation by the tangent use:
//
//      Quat delta = Sophus::SO3<double>::exp(tangent).unit_quaternion();
//      auto new_rotation = delta * old_rotation;
//
template <typename T = double>
HD inline Vector<T, 3> RotatePoint(const Eigen::Quaternion<T>& rotation, const Vector<T, 3>& point,
                                   Matrix<T, 3, 3>* jacobian_rotation  = nullptr,
                                   Matrix<T, 3, 3>* jacobian_point = nullptr)
{
    const Vector<T, 3> rotated_point = rotation * point;

    if (jacobian_rotation)
    {
        *jacobian_rotation = -skew(rotated_point);
    }

    if (jacobian_point)
    {
        *jacobian_point = rotation.matrix();
    }

    return rotated_point;
}

template <typename T = double>
HD inline Vector<T, 3> TransformPoint(const Sophus::SE3<T>& pose, const Vector<T, 3>& point,
                                      Matrix<T, 3, 6>* jacobian_pose  = nullptr,
                                      Matrix<T, 3, 3>* jacobian_point = nullptr)
{
    const Vector<T, 3> rotated_point = pose * point;

    if (jacobian_pose)
    {
        // 1. Translation
        jacobian_pose->template block<3, 3>(0, 0) = Eigen::Matrix<T, 3, 3>::Identity();
        // 2. Rotation
        jacobian_pose->template block<3, 3>(0, 3) = -skew(rotated_point);
    }

    if (jacobian_point)
    {
        *jacobian_point = pose.so3().matrix();
    }

    return rotated_point;
}


template <typename T = double>
HD inline Vector<T, 2> DivideByZ(const Vector<T, 3>& point, Matrix<T, 2, 3>* jacobian_point = nullptr)
{
    auto x                    = point(0);
    auto y                    = point(1);
    auto z                    = point(2);
    auto iz                   = 1 / z;
    const Vector<T, 2> p_by_z = point.template head<2>() * iz;


    if (jacobian_point)
    {
        (*jacobian_point)(0, 0) = iz;
        (*jacobian_point)(0, 1) = 0;
        (*jacobian_point)(0, 2) = -x * iz * iz;

        (*jacobian_point)(1, 0) = 0;
        (*jacobian_point)(1, 1) = iz;
        (*jacobian_point)(1, 2) = -y * iz * iz;
    }

    return p_by_z;
}


// Returns: <Residual, Depth>
template <typename T = double>
inline std::pair<Vector<T, 2>, T> BundleAdjustment(const IntrinsicsPinhole<T>& camera, const Vector<T, 2>& observation,
                                                   const Sophus::SE3<T>& pose, const Vector<T, 3>& point, T weight,
                                                   Matrix<T, 2, 6>* jacobian_pose  = nullptr,
                                                   Matrix<T, 2, 3>* jacobian_point = nullptr)
{
    Vector<T, 3> p      = pose * point;
    Vector<T, 2> p_by_z = Vec2(p(0) / p(2), p(1) / p(2));

    Vector<T, 2> residual;
    residual(0) = camera.fx * p_by_z(0) + camera.s * p_by_z(1) + camera.cx - observation(0);
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
        // 1. Translation

        // division by z
        (*jacobian_pose)(0, 0) = zinv;
        (*jacobian_pose)(0, 1) = 0;
        (*jacobian_pose)(0, 2) = -x * zzinv;
        (*jacobian_pose)(1, 0) = 0;
        (*jacobian_pose)(1, 1) = zinv;
        (*jacobian_pose)(1, 2) = -y * zzinv;

        // multiplication by K
        (*jacobian_pose)(0, 0) = (*jacobian_pose)(0, 0) * camera.fx + (*jacobian_pose)(1, 0) * camera.s;
        (*jacobian_pose)(0, 1) = (*jacobian_pose)(0, 1) * camera.fx + (*jacobian_pose)(1, 1) * camera.s;
        (*jacobian_pose)(0, 2) = (*jacobian_pose)(0, 2) * camera.fx + (*jacobian_pose)(1, 2) * camera.s;
        (*jacobian_pose)(1, 0) *= camera.fy;
        (*jacobian_pose)(1, 1) *= camera.fy;
        (*jacobian_pose)(1, 2) *= camera.fy;


        // 2. Rotation

        // division by z
        (*jacobian_pose)(0, 3) = -y * x * zzinv;
        (*jacobian_pose)(0, 4) = (1 + (x * x) * zzinv);
        (*jacobian_pose)(0, 5) = -y * zinv;
        (*jacobian_pose)(1, 3) = (-1 - (y * y) * zzinv);
        (*jacobian_pose)(1, 4) = x * y * zzinv;
        (*jacobian_pose)(1, 5) = x * zinv;

        // multiplication by K
        (*jacobian_pose)(0, 3) = (*jacobian_pose)(0, 3) * camera.fx + (*jacobian_pose)(1, 3) * camera.s;
        (*jacobian_pose)(0, 4) = (*jacobian_pose)(0, 4) * camera.fx + (*jacobian_pose)(1, 4) * camera.s;
        (*jacobian_pose)(0, 5) = (*jacobian_pose)(0, 5) * camera.fx + (*jacobian_pose)(1, 5) * camera.s;
        (*jacobian_pose)(1, 3) *= camera.fy;
        (*jacobian_pose)(1, 4) *= camera.fy;
        (*jacobian_pose)(1, 5) *= camera.fy;

        // 3. Weight
        (*jacobian_pose) *= weight;
    }

    if (jacobian_point)
    {
        auto R = pose.so3().matrix();

        // division by z
        (*jacobian_point)(0, 0) = (R(0, 0) - p_by_z(0) * R(2, 0)) * zinv;
        (*jacobian_point)(0, 1) = (R(0, 1) - p_by_z(0) * R(2, 1)) * zinv;
        (*jacobian_point)(0, 2) = (R(0, 2) - p_by_z(0) * R(2, 2)) * zinv;
        (*jacobian_point)(1, 0) = (R(1, 0) - p_by_z(1) * R(2, 0)) * zinv;
        (*jacobian_point)(1, 1) = (R(1, 1) - p_by_z(1) * R(2, 1)) * zinv;
        (*jacobian_point)(1, 2) = (R(1, 2) - p_by_z(1) * R(2, 2)) * zinv;

        // multiplication by K
        (*jacobian_point)(0, 0) = (*jacobian_point)(0, 0) * camera.fx + (*jacobian_point)(1, 0) * camera.s;
        (*jacobian_point)(0, 1) = (*jacobian_point)(0, 1) * camera.fx + (*jacobian_point)(1, 1) * camera.s;
        (*jacobian_point)(0, 2) = (*jacobian_point)(0, 2) * camera.fx + (*jacobian_point)(1, 2) * camera.s;
        (*jacobian_point)(1, 0) *= camera.fy;
        (*jacobian_point)(1, 1) *= camera.fy;
        (*jacobian_point)(1, 2) *= camera.fy;

        (*jacobian_point) *= weight;
    }
    return {residual, p(2)};
}

// Returns: <Residual, Depth>
template <typename T = double>
inline std::pair<Vec3, double> BundleAdjustmentStereo(const StereoCamera4& camera, const Vec2& observation,
                                                      double observed_stereo_point, const SE3& pose, const Vec3& point,
                                                      double weight, double weight_depth,
                                                      Matrix<double, 3, 6>* jacobian_pose  = nullptr,
                                                      Matrix<double, 3, 3>* jacobian_point = nullptr)
{
    Vec3 p      = pose * point;
    Vec2 p_by_z = Vec2(p(0) / p(2), p(1) / p(2));

    Vec2 projected_point(camera.fx * p_by_z(0) + camera.s * p_by_z(1) + camera.cx, camera.fy * p_by_z(1) + camera.cy);

    Vec3 residual;
    residual.head<2>() = projected_point - observation;


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

        // 1. Translation
        // division by z
        J(0, 0) = zinv;
        J(0, 1) = 0;
        J(0, 2) = -x * zzinv;
        J(1, 0) = 0;
        J(1, 1) = zinv;
        J(1, 2) = -y * zzinv;

        // multiplication by K
        (*jacobian_pose)(0, 0) = (*jacobian_pose)(0, 0) * camera.fx + (*jacobian_pose)(1, 0) * camera.s;
        (*jacobian_pose)(0, 1) = (*jacobian_pose)(0, 1) * camera.fx + (*jacobian_pose)(1, 1) * camera.s;
        (*jacobian_pose)(0, 2) = (*jacobian_pose)(0, 2) * camera.fx + (*jacobian_pose)(1, 2) * camera.s;
        (*jacobian_pose)(1, 0) *= camera.fy;
        (*jacobian_pose)(1, 1) *= camera.fy;
        (*jacobian_pose)(1, 2) *= camera.fy;

        // 2. Rotation
        // division by z
        J(0, 3) = -y * x * zzinv;
        J(0, 4) = (1 + (x * x) * zzinv);
        J(0, 5) = -y * zinv;
        J(1, 3) = (-1 - (y * y) * zzinv);
        J(1, 4) = x * y * zzinv;
        J(1, 5) = x * zinv;

        // multiplication by K
        (*jacobian_pose)(0, 3) = (*jacobian_pose)(0, 3) * camera.fx + (*jacobian_pose)(1, 3) * camera.s;
        (*jacobian_pose)(0, 4) = (*jacobian_pose)(0, 4) * camera.fx + (*jacobian_pose)(1, 4) * camera.s;
        (*jacobian_pose)(0, 5) = (*jacobian_pose)(0, 5) * camera.fx + (*jacobian_pose)(1, 5) * camera.s;
        (*jacobian_pose)(1, 3) *= camera.fy;
        (*jacobian_pose)(1, 4) *= camera.fy;
        (*jacobian_pose)(1, 5) *= camera.fy;

        // depth
        J(2, 0) = -J(0, 0);
        J(2, 1) = -J(0, 1);
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

        // division by z
        J(0, 0) = (R(0, 0) - p_by_z(0) * R(2, 0)) * zinv;
        J(0, 1) = (R(0, 1) - p_by_z(0) * R(2, 1)) * zinv;
        J(0, 2) = (R(0, 2) - p_by_z(0) * R(2, 2)) * zinv;
        J(1, 0) = (R(1, 0) - p_by_z(1) * R(2, 0)) * zinv;
        J(1, 1) = (R(1, 1) - p_by_z(1) * R(2, 1)) * zinv;
        J(1, 2) = (R(1, 2) - p_by_z(1) * R(2, 2)) * zinv;

        // multiplication by K
        (*jacobian_point)(0, 0) = (*jacobian_point)(0, 0) * camera.fx + (*jacobian_point)(1, 0) * camera.s;
        (*jacobian_point)(0, 1) = (*jacobian_point)(0, 1) * camera.fx + (*jacobian_point)(1, 1) * camera.s;
        (*jacobian_point)(0, 2) = (*jacobian_point)(0, 2) * camera.fx + (*jacobian_point)(1, 2) * camera.s;
        (*jacobian_point)(1, 0) *= camera.fy;
        (*jacobian_point)(1, 1) *= camera.fy;
        (*jacobian_point)(1, 2) *= camera.fy;

        // stereo projection
        J(2, 0) = -J(0, 0) - camera.bf * R(2, 0) * zzinv;
        J(2, 1) = -J(0, 1) - camera.bf * R(2, 1) * zzinv;
        J(2, 2) = -J(0, 2) - camera.bf * R(2, 2) * zzinv;

        // weight
        J.row(0) *= weight;
        J.row(1) *= weight;
        J.row(2) *= weight_depth;
    }
    return {residual, p(2)};
}


}  // namespace Saiga
