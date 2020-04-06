/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/pgo/PGOConfig.h"
namespace Saiga
{
namespace Kernel
{
// Works both for SE3 and Sim3
template <typename TransformationType = SE3>
struct PGO
{
    static constexpr int ResCount     = TransformationType::DoF;
    static constexpr int VarCountPose = TransformationType::DoF;

    using T                 = typename TransformationType::Scalar;
    using ResidualType      = Eigen::Matrix<T, ResCount, 1>;
    using ResidualBlockType = Eigen::Matrix<T, VarCountPose, 1>;
    using PoseJacobiType    = Eigen::Matrix<T, ResCount, VarCountPose, Eigen::RowMajor>;
    using PoseDiaBlockType  = Eigen::Matrix<T, VarCountPose, VarCountPose, Eigen::RowMajor>;


    static inline void evaluateResidual(const TransformationType& T_i_j, const TransformationType& T_w_i,
                                        const TransformationType& T_w_j, ResidualType& res, T weight)
    {
        TransformationType T_j_i = T_w_j.inverse() * T_w_i;
        res                      = Sophus::se3_logd(T_i_j * T_j_i) * weight;
    }

    static inline void evaluateResidualAndJacobian(const TransformationType& T_i_j, const TransformationType& T_w_i,
                                                   const TransformationType& T_w_j, ResidualType& res,
                                                   PoseJacobiType& d_res_d_T_w_i, PoseJacobiType& d_res_d_T_w_j,
                                                   T weight)
    {
        Sophus::SE3d T_j_i = T_w_j.inverse() * T_w_i;

        res = Sophus::se3_logd(T_i_j * T_j_i) * weight;

        Sophus::Matrix6d J;
        Sophus::rightJacobianInvSE3Decoupled(res, J);

        Eigen::Matrix3d R = T_w_i.so3().inverse().matrix();

        Sophus::Matrix6d Adj;
        Adj.setZero();
        Adj.topLeftCorner<3, 3>()     = R;
        Adj.bottomRightCorner<3, 3>() = R;


        d_res_d_T_w_i = J * Adj;



        Adj.topRightCorner<3, 3>() = Sophus::SO3d::hat(T_j_i.inverse().translation()) * R;
        d_res_d_T_w_j              = -J * Adj;
    }
};

template <typename T>
struct PGOSim3
{
    using TransformationType = Sophus::DSim3<T>;

    static constexpr int ResCount     = TransformationType::DoF;
    static constexpr int VarCountPose = TransformationType::DoF;

    using ResidualType      = Eigen::Matrix<T, ResCount, 1>;
    using ResidualBlockType = Eigen::Matrix<T, VarCountPose, 1>;
    using PoseJacobiType    = Eigen::Matrix<T, ResCount, VarCountPose, Eigen::RowMajor>;
    using PoseDiaBlockType  = Eigen::Matrix<T, VarCountPose, VarCountPose, Eigen::RowMajor>;


    static inline void evaluateResidual(const TransformationType& T_i_j, const TransformationType& T_w_i,
                                        const TransformationType& T_w_j, ResidualType& res, T weight)
    {
        TransformationType T_j_i = T_w_j.inverse() * T_w_i;
        res                      = Sophus::dsim3_logd(T_i_j * T_j_i) * weight;
    }

    static inline void evaluateResidualAndJacobian(const TransformationType& T_i_j, const TransformationType& T_w_i,
                                                   const TransformationType& T_w_j, ResidualType& res,
                                                   PoseJacobiType& d_res_d_T_w_i, PoseJacobiType& d_res_d_T_w_j,
                                                   T weight)
    {
        TransformationType T_j_i = T_w_j.inverse() * T_w_i;
        res                      = Sophus::dsim3_logd(T_i_j * T_j_i) * weight;

        Sophus::Matrix6d J;
        Sophus::rightJacobianInvSE3Decoupled(res.template head<6>(), J);
        J.template topLeftCorner<3, 3>() *= exp(res(6));

        Eigen::Matrix3d R = T_w_i.se3().so3().inverse().matrix();

        Sophus::Matrix6d Adj;
        Adj.setZero();
        Adj.topLeftCorner<3, 3>()     = (1.0 / T_w_i.scale()) * R;
        Adj.bottomRightCorner<3, 3>() = R;


        d_res_d_T_w_i.setZero();
        d_res_d_T_w_i.template topLeftCorner<6, 6>() = J * Adj;
        d_res_d_T_w_i(6, 6)                          = 1.0;



        Adj.topRightCorner<3, 3>() = Sophus::SO3d::hat(T_j_i.se3().inverse().translation()) * R;

        d_res_d_T_w_j.setZero();
        d_res_d_T_w_j.template topLeftCorner<6, 6>() = -J * Adj;
        d_res_d_T_w_j.template block<3, 1>(0, 6)     = J.topLeftCorner<3, 3>() * T_j_i.inverse().se3().translation();
        //        d_res_d_T_w_j *= exp(res(6));
        d_res_d_T_w_j(6, 6) = -1.0;
    }
};



}  // namespace Kernel
}  // namespace Saiga
