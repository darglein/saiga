﻿/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/vision/kernels/BA.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"

namespace Saiga
{
#ifndef WIN32

Vec2 BundleAdjustmentVerbose(const IntrinsicsPinholed& camera, const Vec2& observation, const SE3& pose,
                             const Vec3& point, double weight, Matrix<double, 2, 6>* jacobian_pose = nullptr,
                             Matrix<double, 2, 3>* jacobian_point = nullptr)
{
    Vec3 p      = pose * point;
    Vec2 p_by_z = Vec2(p(0) / p(2), p(1) / p(2));

    Vec2 residual;
    residual(0) = camera.fx * p_by_z(0) + camera.s * p_by_z(1) + camera.cx - observation(0);
    residual(1) = camera.fy * p_by_z(1) + camera.cy - observation(1);
    residual *= weight;
    auto x = p(0);
    auto y = p(1);
    auto z = p(2);

    if (jacobian_pose)
    {
        // 1. Translation
        Matrix<double, 3, 3> J = Matrix<double, 3, 3>::Identity();

        // division by z
        (*jacobian_pose)(0, 0) = (J(0, 0) * z - x * J(2, 0)) / (z * z);
        (*jacobian_pose)(0, 1) = (J(0, 1) * z - x * J(2, 1)) / (z * z);
        (*jacobian_pose)(0, 2) = (J(0, 2) * z - x * J(2, 2)) / (z * z);
        (*jacobian_pose)(1, 0) = (J(1, 0) * z - y * J(2, 0)) / (z * z);
        (*jacobian_pose)(1, 1) = (J(1, 1) * z - y * J(2, 1)) / (z * z);
        (*jacobian_pose)(1, 2) = (J(1, 2) * z - y * J(2, 2)) / (z * z);

        // multiplication by K
        (*jacobian_pose)(0, 0) = (*jacobian_pose)(0, 0) * camera.fx + (*jacobian_pose)(1, 0) * camera.s;
        (*jacobian_pose)(0, 1) = (*jacobian_pose)(0, 1) * camera.fx + (*jacobian_pose)(1, 1) * camera.s;
        (*jacobian_pose)(0, 2) = (*jacobian_pose)(0, 2) * camera.fx + (*jacobian_pose)(1, 2) * camera.s;
        (*jacobian_pose)(1, 0) *= camera.fy;
        (*jacobian_pose)(1, 1) *= camera.fy;
        (*jacobian_pose)(1, 2) *= camera.fy;


        // 2. Rotation
        J(0, 0) = 0;
        J(0, 1) = z;
        J(0, 2) = -y;
        J(1, 0) = -z;
        J(1, 1) = 0;
        J(1, 2) = x;
        J(2, 0) = y;
        J(2, 1) = -x;
        J(2, 2) = 0;

        // division by z
        (*jacobian_pose)(0, 3) = (J(0, 0) * z - x * J(2, 0)) / (z * z);
        (*jacobian_pose)(0, 4) = (J(0, 1) * z - x * J(2, 1)) / (z * z);
        (*jacobian_pose)(0, 5) = (J(0, 2) * z - x * J(2, 2)) / (z * z);
        (*jacobian_pose)(1, 3) = (J(1, 0) * z - y * J(2, 0)) / (z * z);
        (*jacobian_pose)(1, 4) = (J(1, 1) * z - y * J(2, 1)) / (z * z);
        (*jacobian_pose)(1, 5) = (J(1, 2) * z - y * J(2, 2)) / (z * z);

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
        (*jacobian_point)(0, 0) = (R(0, 0) * z - x * R(2, 0)) / (z * z);
        (*jacobian_point)(0, 1) = (R(0, 1) * z - x * R(2, 1)) / (z * z);
        (*jacobian_point)(0, 2) = (R(0, 2) * z - x * R(2, 2)) / (z * z);
        (*jacobian_point)(1, 0) = (R(1, 0) * z - y * R(2, 0)) / (z * z);
        (*jacobian_point)(1, 1) = (R(1, 1) * z - y * R(2, 1)) / (z * z);
        (*jacobian_point)(1, 2) = (R(1, 2) * z - y * R(2, 2)) / (z * z);

        // multiplication by K
        (*jacobian_point)(0, 0) = (*jacobian_point)(0, 0) * camera.fx + (*jacobian_point)(1, 0) * camera.s;
        (*jacobian_point)(0, 1) = (*jacobian_point)(0, 1) * camera.fx + (*jacobian_point)(1, 1) * camera.s;
        (*jacobian_point)(0, 2) = (*jacobian_point)(0, 2) * camera.fx + (*jacobian_point)(1, 2) * camera.s;
        (*jacobian_point)(1, 0) *= camera.fy;
        (*jacobian_point)(1, 1) *= camera.fy;
        (*jacobian_point)(1, 2) *= camera.fy;

        (*jacobian_point) *= weight;
    }
    return residual;
}



TEST(NumericDerivative, BundleAdjustment)
{
    Random::setSeed(49367346);
    SE3 pose_c_w = Random::randomSE3();
    Vec3 wp      = Vec3::Random();
    IntrinsicsPinholed intr;
    intr.coeffs(Vec5::Random());

    Vec2 projection  = intr.project(pose_c_w * wp);
    Vec2 observation = projection + Vec2::Random() * 0.1;

    double weight = 6;
    Matrix<double, 2, 6> J_pose_1, J_pose_ref, J_pose_3;
    Matrix<double, 2, 3> J_point_1, J_point_ref, J_point_3;
    Vec2 res1, res_ref, res3;

    res1 = BundleAdjustment(intr, observation, pose_c_w, wp, weight, &J_pose_1, &J_point_1).first;
    res3 = BundleAdjustmentVerbose(intr, observation, pose_c_w, wp, weight, &J_pose_3, &J_point_3);

    {
        Vec6 eps = Vec6::Zero();
        res_ref  = EvaluateNumeric(
            [=](auto p) {
                auto pose_c_w_new = Sophus::se3_expd(p) * pose_c_w;
                //                auto pose_c_w_new = pose_c_w;
                //                Sophus::decoupled_inc(p, pose_c_w_new);

                return BundleAdjustment(intr, observation, pose_c_w_new, wp, weight).first;
            },
            eps, &J_pose_ref);
    }
    {
        res_ref = EvaluateNumeric(
            [=](auto p) { return BundleAdjustment(intr, observation, pose_c_w, p, weight).first; }, wp, &J_point_ref);
    }

    ExpectCloseRelative(res_ref, res1, 1e-5);
    ExpectCloseRelative(res_ref, res3, 1e-5);
    ExpectCloseRelative(J_point_ref, J_point_3, 1e-5);
    ExpectCloseRelative(J_pose_ref, J_pose_3, 1e-5);
    ExpectCloseRelative(J_pose_ref, J_pose_1, 1e-5);
    ExpectCloseRelative(J_point_ref, J_point_1, 1e-5);
}

TEST(NumericDerivative, BundleAdjustmentStereo)
{
    SE3 pose_c_w = Random::randomSE3();
    Vec3 wp      = Vec3::Random();
    StereoCamera4 intr;
    intr.coeffs(Vec6::Random());

    double stereo_point = Random::sampleDouble(-1, 1);

    Vec2 projection  = intr.project(pose_c_w * wp);
    Vec2 observation = projection + Vec2::Random() * 0.1;

    double weight       = 6;
    double weight_depth = 0.7;
    Matrix<double, 3, 6> J_pose_1, J_pose_ref;
    Matrix<double, 3, 3> J_point_1, J_point_ref;
    Vec3 res1, res_ref;

    res1 = BundleAdjustmentStereo(intr, observation, stereo_point, pose_c_w, wp, weight, weight_depth, &J_pose_1,
                                  &J_point_1)
               .first;

    {
        Vec6 eps = Vec6::Zero();
        res_ref  = EvaluateNumeric(
            [=](auto p) {
                auto pose_c_w_new = Sophus::se3_expd(p) * pose_c_w;
                return BundleAdjustmentStereo(intr, observation, stereo_point, pose_c_w_new, wp, weight, weight_depth)
                    .first;
            },
            eps, &J_pose_ref);
    }
    {
        res_ref = EvaluateNumeric(
            [=](auto p) {
                return BundleAdjustmentStereo(intr, observation, stereo_point, pose_c_w, p, weight, weight_depth).first;
            },
            wp, &J_point_ref);
    }

    ExpectCloseRelative(res_ref, res1, 1e-5);
    ExpectCloseRelative(J_point_ref, J_point_1, 1e-5);
    ExpectCloseRelative(J_pose_ref, J_pose_1, 1e-5);
}


TEST(NumericDerivative, Distortion)
{
    Vector<double, 8> c;
    c.setRandom();
    Distortion d(c);

    Vec2 p = Vec2::Random();
    Matrix<double, 2, 2> J1, J2;

    Vec2 res1 = distortNormalizedPoint(p, d, &J1);
    Vec2 res2 = EvaluateNumeric([&](auto p) { return distortNormalizedPoint(p, d); }, p, &J2, 1e-8);

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J1, J2, 1e-5);
}


Vec2 BundleAdjustmentDistortionVerbose(const IntrinsicsPinholed& camera, const Distortion& distortion,
                                       const Vec2& observation, const SE3& pose, const Vec3& point, double weight,
                                       Matrix<double, 2, 6>* jacobian_pose  = nullptr,
                                       Matrix<double, 2, 3>* jacobian_point = nullptr)
{
    // 1. Pose

    Matrix<double, 3, 6> J_pose;
    Matrix<double, 3, 3> J_point;
    const Vec3 p = TransformPoint(pose, point, &J_pose, &J_point);

    // 2. Divide by z
    Matrix<double, 2, 3> J_point_div;
    const Vec2 p_by_z = DivideByZ(p, &J_point_div);

    // 3. Distortion
    Mat2 J_dis;
    const Vec2 distorted_point =
        distortNormalizedPoint(p_by_z, distortion, (jacobian_pose || jacobian_point) ? &J_dis : nullptr);

    // 4. K
    Mat2 J_K;
    const Vec2 image_point =
        camera.normalizedToImage(distorted_point, (jacobian_pose || jacobian_point) ? &J_K : nullptr);

    // 5. residual
    Vec2 residual = image_point - observation;
    residual *= weight;

    if (jacobian_pose)
    {
        *jacobian_pose = weight * J_K * J_dis * J_point_div * J_pose;
    }

    if (jacobian_point)
    {
        *jacobian_point = weight * J_K * J_dis * J_point_div * J_point;
    }
    return residual;
}


TEST(NumericDerivative, BundleAdjustmentDistortion)
{
    Random::setSeed(49367346);
    SE3 pose_c_w = Random::randomSE3();
    Vec3 wp      = Vec3::Random();
    IntrinsicsPinholed intr;
    intr.coeffs(Vec5::Random());

    Vector<double, 8> c;
    c.setRandom();
    c *= 0.01;
    Distortion d(c);
    //    d = Distortion();

    Vec2 projection  = intr.project(pose_c_w * wp);
    Vec2 observation = projection + Vec2::Random() * 0.1;

    double weight = 6;
    Matrix<double, 2, 6> J_pose_ref, J_pose_3;
    Matrix<double, 2, 3> J_point_ref, J_point_3;
    Vec2 res_ref, res3;

    res3 = BundleAdjustmentDistortionVerbose(intr, d, observation, pose_c_w, wp, weight, &J_pose_3, &J_point_3);

    {
        Vec6 eps = Vec6::Zero();
        res_ref  = EvaluateNumeric(
            [=](auto p) {
                auto pose_c_w_new = Sophus::se3_expd(p) * pose_c_w;
                return BundleAdjustmentDistortionVerbose(intr, d, observation, pose_c_w_new, wp, weight);
            },
            eps, &J_pose_ref);
    }
    {
        res_ref = EvaluateNumeric(
            [=](auto p) { return BundleAdjustmentDistortionVerbose(intr, d, observation, pose_c_w, p, weight); }, wp,
            &J_point_ref);
    }

    ExpectCloseRelative(res_ref, res3, 1e-5);
    ExpectCloseRelative(J_point_ref, J_point_3, 1e-5);
    ExpectCloseRelative(J_pose_ref, J_pose_3, 1e-5);
}

#endif
}  // namespace Saiga