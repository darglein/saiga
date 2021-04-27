/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/pgo/PGOConfig.h"
namespace Saiga
{
// using left multiplied se3_expd
inline Sophus::Vector6d relPoseError(const Sophus::SE3d& T_i_j, const Sophus::SE3d& T_w_i, const Sophus::SE3d& T_w_j,
                                     double weight_rotation, double weight_translation,
                                     Sophus::Matrix6d* d_res_d_T_w_i = nullptr,
                                     Sophus::Matrix6d* d_res_d_T_w_j = nullptr)
{
    Sophus::SE3d T_j_i        = T_w_j.inverse() * T_w_i;
    Sophus::Vector6d res      = Sophus::se3_logd(T_i_j * T_j_i);
    Sophus::Vector6d residual = res;
    residual.head<3>() *= weight_translation;
    residual.tail<3>() *= weight_rotation;

    if (d_res_d_T_w_i || d_res_d_T_w_j)
    {
        Sophus::Matrix6d J;
        Sophus::rightJacobianInvSE3Decoupled(res, J);

        J.topLeftCorner<3, 3>() *= weight_translation;
        J.bottomRightCorner<3, 3>() *= weight_rotation;

        Eigen::Matrix3d R = T_w_i.so3().inverse().matrix();

        Sophus::Matrix6d Adj;
        Adj.setZero();
        Adj.topLeftCorner<3, 3>()     = R;
        Adj.bottomRightCorner<3, 3>() = R;
        Adj.topRightCorner<3, 3>()    = Sophus::SO3d::hat(T_w_i.inverse().translation()) * R;

        if (d_res_d_T_w_i)
        {
            *d_res_d_T_w_i = J * Adj;
        }

        if (d_res_d_T_w_j)
        {
            if (d_res_d_T_w_i)
            {
                *d_res_d_T_w_j = -(*d_res_d_T_w_i);
            }
            else
            {
                *d_res_d_T_w_j = -J * Adj;
            }
        }
    }

    return residual;
}



inline Sophus::Vector6d relPoseErrorView(const Sophus::SE3d& T_i_j, const Sophus::SE3d& T_w_i,
                                         const Sophus::SE3d& T_w_j, double weight_rotation, double weight_translation,
                                         Sophus::Matrix6d* d_res_d_T_w_i = nullptr,
                                         Sophus::Matrix6d* d_res_d_T_w_j = nullptr)
{
    Sophus::SE3d T_j_i        = T_w_j * T_w_i.inverse();
    Sophus::Vector6d res      = Sophus::se3_logd(T_i_j * T_j_i);
    Sophus::Vector6d residual = res;
    residual.head<3>() *= weight_translation;
    residual.tail<3>() *= weight_rotation;


    if (d_res_d_T_w_i || d_res_d_T_w_j)
    {
        Sophus::Matrix6d J;
        Sophus::rightJacobianInvSE3Decoupled(res, J);

        J.topLeftCorner<3, 3>() *= weight_translation;
        J.bottomRightCorner<3, 3>() *= weight_rotation;

        if (d_res_d_T_w_i)
        {
            d_res_d_T_w_i->setZero();
            *d_res_d_T_w_i = -J;
        }

        if (d_res_d_T_w_j)
        {
            Eigen::Matrix3d R = T_j_i.so3().inverse().matrix();
            Sophus::Matrix6d Adj;
            Adj.setZero();
            Adj.topLeftCorner<3, 3>()     = R;
            Adj.bottomRightCorner<3, 3>() = R;
            Adj.topRightCorner<3, 3>()    = Sophus::SO3d::hat(T_j_i.inverse().translation()) * R;
            d_res_d_T_w_j->setZero();
            (*d_res_d_T_w_j) = J * Adj;
        }
    }

    return residual;
}



inline Sophus::Vector7d relPoseError(const Sophus::DSim3<double>& T_i_j, const Sophus::DSim3<double>& T_w_i,
                                     const Sophus::DSim3<double>& T_w_j, double weight_rotation,
                                     double weight_translation, Sophus::Matrix7d* d_res_d_T_w_i = nullptr,
                                     Sophus::Matrix7d* d_res_d_T_w_j = nullptr)
{
    Sophus::DSim3<double> T_j_i = T_w_j.inverse() * T_w_i;
    Sophus::Vector7d res        = Sophus::dsim3_logd(T_i_j * T_j_i);
    Sophus::Vector7d residual   = res;

    residual.head<3>() *= weight_translation;
    residual.segment<3>(3) *= weight_rotation;

    if (d_res_d_T_w_i || d_res_d_T_w_j)
    {
        Sophus::Matrix7d J;
        Sophus::rightJacobianInvDSim3Decoupled(res, J);

        J.topLeftCorner<3, 3>() *= weight_translation;
        J.block<3, 3>(3, 3) *= weight_rotation;

        Eigen::Matrix3d R = T_w_i.se3().so3().inverse().matrix();

        Sophus::Matrix7d Adj;
        Adj.setZero();
        Adj.topLeftCorner<3, 3>() = (1.0 / T_w_i.scale()) * R;
        Adj.block<3, 3>(3, 3)     = R;

        Adj.block<3, 3>(0, 3) = Sophus::SO3d::hat(T_w_i.inverse().se3().translation()) * R;
        Adj.block<3, 1>(0, 6) = -T_w_i.inverse().se3().translation();

        Adj(6, 6) = 1;


        if (d_res_d_T_w_i)
        {
            *d_res_d_T_w_i = J * Adj;
        }

        if (d_res_d_T_w_j)
        {
            if (d_res_d_T_w_i)
            {
                *d_res_d_T_w_j = -(*d_res_d_T_w_i);
            }
            else
            {
                *d_res_d_T_w_j = -J * Adj;
            }
        }
    }

    return residual;
}

}  // namespace Saiga
