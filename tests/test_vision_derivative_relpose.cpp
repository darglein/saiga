/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/vision/kernels/PGO.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"

namespace Saiga
{
//
// Source:
// https://gitlab.com/VladyslavUsenko/basalt/-/blob/24e378a7a100d7d6f5133b34e16a09bb0cc98459/include/basalt/utils/nfr.h#L43-73
//
// using the decoupled inc
inline Sophus::Vector6d relPoseErrorDecoupled(const Sophus::SE3d& T_i_j, const Sophus::SE3d& T_w_i,
                                              const Sophus::SE3d& T_w_j, Sophus::Matrix6d* d_res_d_T_w_i = nullptr,
                                              Sophus::Matrix6d* d_res_d_T_w_j = nullptr)
{
    Sophus::SE3d T_j_i   = T_w_j.inverse() * T_w_i;
    Sophus::Vector6d res = Sophus::se3_logd(T_i_j * T_j_i);

    if (d_res_d_T_w_i || d_res_d_T_w_j)
    {
        Sophus::Matrix6d J;
        Sophus::rightJacobianInvSE3Decoupled(res, J);

        Eigen::Matrix3d R = T_w_i.so3().inverse().matrix();

        Sophus::Matrix6d Adj;
        Adj.setZero();
        Adj.topLeftCorner<3, 3>()     = R;
        Adj.bottomRightCorner<3, 3>() = R;

        if (d_res_d_T_w_i)
        {
            *d_res_d_T_w_i = J * Adj;
        }

        if (d_res_d_T_w_j)
        {
            Adj.topRightCorner<3, 3>() = Sophus::SO3d::hat(T_j_i.inverse().translation()) * R;
            *d_res_d_T_w_j             = -J * Adj;
        }
    }

    return res;
}


TEST(DerivativeRelpose, RelativePoseDecoupled)
{
    Random::setSeed(903476346);
    srand(976157);

    SE3 pose_w_i = Random::randomSE3();
    SE3 pose_w_j = Random::randomSE3();
    SE3 pose_i_j = Sophus::se3_expd(Sophus::Vector6d::Random() / 100) * pose_w_i.inverse() * pose_w_j;

    Matrix<double, 6, 6> J_pose_i_1, J_pose_j_1, J_pose_i_2, J_pose_j_2;
    J_pose_i_1.setZero();
    J_pose_j_1.setZero();
    J_pose_i_2.setZero();
    J_pose_j_2.setZero();
    Vec6 res1, res2;

    res1 = relPoseErrorDecoupled(pose_i_j, pose_w_i, pose_w_j, &J_pose_i_1, &J_pose_j_1);

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                //                auto pose_w_i_new = Sophus::se3_expd(p) * pose_w_i;

                auto pose_w_i_new = pose_w_i;
                Sophus::decoupled_inc(p, pose_w_i_new);

                return relPoseErrorDecoupled(pose_i_j, pose_w_i_new, pose_w_j);
            },
            eps, &J_pose_i_2);
    }

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                //                auto pose_w_j_new = Sophus::se3_expd(p) * pose_w_j;
                auto pose_w_j_new = pose_w_j;
                Sophus::decoupled_inc(p, pose_w_j_new);

                return relPoseErrorDecoupled(pose_i_j, pose_w_i, pose_w_j_new);
            },
            eps, &J_pose_j_2);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_pose_i_1, J_pose_i_2, 1e-5);
    ExpectCloseRelative(J_pose_j_1, J_pose_j_2, 1e-5);
}


TEST(DerivativeRelpose, RelativePose)
{
    SE3 pose_w_i = Random::randomSE3();
    SE3 pose_w_j = Random::randomSE3();
    SE3 pose_i_j = Sophus::se3_expd(Sophus::Vector6d::Random() / 100) * pose_w_i.inverse() * pose_w_j;

    double weight_translation = 2;
    double weight_rotation    = 5;

    Matrix<double, 6, 6> J_pose_i_1, J_pose_j_1, J_pose_i_2, J_pose_j_2;
    J_pose_i_1.setZero();
    J_pose_j_1.setZero();
    J_pose_i_2.setZero();
    J_pose_j_2.setZero();
    Vec6 res1, res2;

    res1 = relPoseError(pose_i_j, pose_w_i, pose_w_j, weight_rotation, weight_translation, &J_pose_i_1, &J_pose_j_1);

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto pose_w_i_new = Sophus::se3_expd(p) * pose_w_i;

                //                auto pose_w_i_new = pose_w_i;
                //                Sophus::decoupled_inc(p, pose_w_i_new);

                return relPoseError(pose_i_j, pose_w_i_new, pose_w_j, weight_rotation, weight_translation);
            },
            eps, &J_pose_i_2);
    }

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto pose_w_j_new = Sophus::se3_expd(p) * pose_w_j;
                //                auto pose_w_j_new = pose_w_j;
                //                Sophus::decoupled_inc(p, pose_w_j_new);

                return relPoseError(pose_i_j, pose_w_i, pose_w_j_new, weight_rotation, weight_translation);
            },
            eps, &J_pose_j_2);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_pose_i_1, J_pose_i_2, 1e-5);
    ExpectCloseRelative(J_pose_j_1, J_pose_j_2, 1e-5);
}



TEST(DerivativeRelpose, RelativePoseView)
{
    SE3 pose_w_i = Random::randomSE3();
    SE3 pose_w_j = Random::randomSE3();
    SE3 pose_i_j = Sophus::se3_expd(Sophus::Vector6d::Random() / 10) * pose_w_j * pose_w_i.inverse();
    pose_i_j     = pose_i_j.inverse();

    double weight_translation = 2;
    double weight_rotation    = 5;
    Matrix<double, 6, 6> J_pose_i_1, J_pose_j_1, J_pose_i_2, J_pose_j_2;
    J_pose_i_1.setZero();
    J_pose_j_1.setZero();
    J_pose_i_2.setZero();
    J_pose_j_2.setZero();
    Vec6 res1, res2;

    res1 =
        relPoseErrorView(pose_i_j, pose_w_i, pose_w_j, weight_rotation, weight_translation, &J_pose_i_1, &J_pose_j_1);

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto pose_w_i_new = Sophus::se3_expd(p) * pose_w_i;

                //                auto pose_w_i_new = pose_w_i;
                //                Sophus::decoupled_inc(p, pose_w_i_new);

                return relPoseErrorView(pose_i_j, pose_w_i_new, pose_w_j, weight_rotation, weight_translation);
            },
            eps, &J_pose_i_2);
    }

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto pose_w_j_new = Sophus::se3_expd(p) * pose_w_j;
                //                auto pose_w_j_new = pose_w_j;
                //                Sophus::decoupled_inc(p, pose_w_j_new);

                return relPoseErrorView(pose_i_j, pose_w_i, pose_w_j_new, weight_rotation, weight_translation);
            },
            eps, &J_pose_j_2);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_pose_i_1, J_pose_i_2, 1e-5);
    ExpectCloseRelative(J_pose_j_1, J_pose_j_2, 1e-5);
}


inline Sophus::Vector7d relPoseErrorDecoupled(const Sophus::DSim3<double>& T_i_j, const Sophus::DSim3<double>& T_w_i,
                                              const Sophus::DSim3<double>& T_w_j,
                                              Sophus::Matrix7d* d_res_d_T_w_i = nullptr,
                                              Sophus::Matrix7d* d_res_d_T_w_j = nullptr)
{
    Sophus::DSim3<double> T_j_i = T_w_j.inverse() * T_w_i;
    Sophus::Vector7d res        = Sophus::dsim3_logd(T_i_j * T_j_i);

    if (d_res_d_T_w_i || d_res_d_T_w_j)
    {
        Sophus::Matrix7d J;
        Sophus::rightJacobianInvDSim3Decoupled(res, J);

        Eigen::Matrix3d R = T_w_i.se3().so3().inverse().matrix();

        Sophus::Matrix7d Adj;
        Adj.setZero();
        Adj.topLeftCorner<3, 3>() = (1.0 / T_w_i.scale()) * R;
        Adj.block<3, 3>(3, 3)     = R;


        Adj(6, 6) = 1;

        if (d_res_d_T_w_i)
        {
            (*d_res_d_T_w_i) = J * Adj;
        }

        if (d_res_d_T_w_j)
        {
            Adj.block<3, 3>(0, 3) = Sophus::SO3d::hat(T_j_i.inverse().se3().translation()) * R;
            Adj.block<3, 1>(0, 6) = -T_j_i.inverse().se3().translation();
            (*d_res_d_T_w_j)      = -J * Adj;
        }
    }

    return res;
}


TEST(DerivativeRelpose, RelativePoseSim3Decoupled)
{
    Sophus::DSim3d pose_w_j = Random::randomDSim3();
    Sophus::DSim3d pose_w_i = Random::randomDSim3();

    pose_w_i.scale()        = 0.7;
    pose_w_j.scale()        = 1.0;
    Sophus::DSim3d pose_i_j = Sophus::dsim3_expd(Sophus::Vector7d::Random() / 100) * pose_w_i.inverse() * pose_w_j;

    Matrix<double, 7, 7> J_pose_i_1, J_pose_j_1, J_pose_i_2, J_pose_j_2;
    J_pose_i_1.setZero();
    J_pose_j_1.setZero();
    J_pose_i_2.setZero();
    J_pose_j_2.setZero();
    Vec7 res1, res2;

    res1 = relPoseErrorDecoupled(pose_i_j, pose_w_i, pose_w_j, &J_pose_i_1, &J_pose_j_1);

    {
        Vec7 eps = Vec7::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                //                auto pose_w_i_new = Sophus::dsim3_expd(p) * pose_w_i;

                auto pose_w_i_new = pose_w_i;
                decoupled_inc(p, pose_w_i_new);

                return relPoseErrorDecoupled(pose_i_j, pose_w_i_new, pose_w_j);
            },
            eps, &J_pose_i_2, 1e-4);
    }

    {
        Vec7 eps = Vec7::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                //                auto pose_w_j_new = Sophus::dsim3_expd(p) * pose_w_j;
                auto pose_w_j_new = pose_w_j;
                decoupled_inc(p, pose_w_j_new);

                return relPoseErrorDecoupled(pose_i_j, pose_w_i, pose_w_j_new);
            },
            eps, &J_pose_j_2, 1e-4);
    }

    ExpectCloseRelative(res1, res2, 1e-10);
    ExpectCloseRelative(J_pose_i_1, J_pose_i_2, 1e-8);
    ExpectCloseRelative(J_pose_j_1, J_pose_j_2, 1e-8);
}



TEST(DerivativeRelpose, RelativePoseSim3)
{
    Sophus::DSim3d pose_w_j = Random::randomDSim3();
    Sophus::DSim3d pose_w_i = Random::randomDSim3();

    pose_w_i.scale()        = 0.7;
    pose_w_j.scale()        = 1.0;
    Sophus::DSim3d pose_i_j = Sophus::dsim3_expd(Sophus::Vector7d::Random() / 100) * pose_w_i.inverse() * pose_w_j;

    double weight_translation = 2;
    double weight_rotation    = 5;
    Matrix<double, 7, 7> J_pose_i_1, J_pose_j_1, J_pose_i_2, J_pose_j_2;
    J_pose_i_1.setZero();
    J_pose_j_1.setZero();
    J_pose_i_2.setZero();
    J_pose_j_2.setZero();
    Vec7 res1, res2;

    res1 = relPoseError(pose_i_j, pose_w_i, pose_w_j, weight_rotation, weight_translation, &J_pose_i_1, &J_pose_j_1);

    {
        Vec7 eps = Vec7::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto pose_w_i_new = Sophus::dsim3_expd(p) * pose_w_i;

                //                auto pose_w_i_new = pose_w_i;
                //                decoupled_inc(p, pose_w_i_new);

                return relPoseError(pose_i_j, pose_w_i_new, pose_w_j, weight_rotation, weight_translation);
            },
            eps, &J_pose_i_2, 1e-4);
    }

    {
        Vec7 eps = Vec7::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto pose_w_j_new = Sophus::dsim3_expd(p) * pose_w_j;
                //                auto pose_w_j_new = pose_w_j;
                //                decoupled_inc(p, pose_w_j_new);

                return relPoseError(pose_i_j, pose_w_i, pose_w_j_new, weight_rotation, weight_translation);
            },
            eps, &J_pose_j_2, 1e-4);
    }

    ExpectCloseRelative(res1, res2, 1e-10);
    ExpectCloseRelative(J_pose_i_1, J_pose_i_2, 1e-8);
    ExpectCloseRelative(J_pose_j_1, J_pose_j_2, 1e-8);
}


// for smooth pose estimation during slam.
inline Sophus::Vector6d SmoothPose(const Sophus::SE3d& T_w_i, const Sophus::SE3d& T_w_j, double weight_rotation,
                                   double weight_translation, Sophus::Matrix6d* d_res_d_T_w_i = nullptr)
{
    Sophus::SE3d T_j_i   = T_w_j.inverse() * T_w_i;
    Sophus::Vector6d res = Sophus::se3_logd(T_j_i);

    Vec6 residual = res;
    residual.template segment<3>(0) *= (weight_translation);
    residual.template segment<3>(3) *= (weight_rotation);

    if (d_res_d_T_w_i)
    {
        Sophus::Matrix6d J;
        Sophus::rightJacobianInvSE3Decoupled(res, J);

        Eigen::Matrix3d R = T_w_i.so3().inverse().matrix();

        Sophus::Matrix6d Adj;
        Adj.setZero();
        Adj.topLeftCorner<3, 3>()     = R * weight_translation;
        Adj.bottomRightCorner<3, 3>() = R * weight_rotation;
        Adj.topRightCorner<3, 3>()    = Sophus::SO3d::hat(T_w_i.inverse().translation()) * R * weight_translation;

        *d_res_d_T_w_i = J * Adj;
    }

    return residual;
}


TEST(DerivativeRelpose, SmoothPose)
{
    SE3 pose_w_i = Random::randomSE3();
    SE3 pose_w_j = Sophus::se3_expd(Sophus::Vector6d::Random() / 10) * pose_w_i;

    double wr = 50;
    double wt = 75;

    Matrix<double, 6, 6> J_pose_i_1, J_pose_i_2;
    J_pose_i_1.setZero();
    J_pose_i_2.setZero();
    Vec6 res1, res2;

    res1 = SmoothPose(pose_w_i, pose_w_j, wr, wt, &J_pose_i_1);

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto pose_w_i_new = Sophus::se3_expd(p) * pose_w_i;
                return SmoothPose(pose_w_i_new, pose_w_j, wr, wt);
            },
            eps, &J_pose_i_2);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_pose_i_1, J_pose_i_2, 1e-5);
}


}  // namespace Saiga
