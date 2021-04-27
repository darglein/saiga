/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/kernels/BA.h"

#include "ceres/autodiff_cost_function.h"

namespace Saiga
{
struct CostSmoothBA
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Helper function to simplify the "add residual" part for creating ceres problems
    using CostType = CostSmoothBA;
    // Note: The first number is the number of residuals
    //       The following number sthe size of the residual blocks (without local parametrization)
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 6, 7, 7, 7>;
    template <typename... Types>
    static CostFunctionType* create(const Types&... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _extrinsics1, const T* const _extrinsics2, const T* const _extrinsics3,
                    T* _residuals) const
    {
        using SE3  = Sophus::SE3<T>;
        using Vec6 = Eigen::Matrix<T, 6, 1>;

        Eigen::Map<SE3 const> const p1(_extrinsics1);
        Eigen::Map<SE3 const> const p2(_extrinsics2);
        Eigen::Map<SE3 const> const p3(_extrinsics3);
        Eigen::Map<Vec6> residual(_residuals);


        //        {
        //            auto t1 = p1.inverse().translation();
        //            auto t2 = p2.inverse().translation();
        //            auto t3 = p3.inverse().translation();

        //            residual.setZero();

        //            residual.template segment<3>(0) = (t2-t1) - (t3-t2);
        //        }
        auto rel12 = (p2 * p1.inverse());
        auto rel23 = (p3 * p2.inverse());

        //        auto rel12 = (p1.inverse() * p2);
        //        auto rel23 = (p2.inverse() * p3);


        //        Vec6 t = (rel23 *rel12.inverse()).log();
        //        residual = t;
        Vec6 v1  = rel12.log();
        Vec6 v2  = rel23.log();
        residual = weight * (v1 - v2);

        return true;
    }

    CostSmoothBA(double weight = 1) : weight(weight) {}

    double weight;
};



struct CostRelPose
{
    CostRelPose(const SE3& T_i_j, double weight_rotation, double weight_translation)
        : T_i_j_(T_i_j), weight_rotation(weight_rotation), weight_translation(weight_translation)
    {
    }

    using CostType         = CostRelPose;
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 6, 7, 7>;
    template <typename... Types>
    static CostFunctionType* create(const Types&... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const T_w_i_ptr, const T* const T_w_j_ptr, T* residual_ptr) const
    {
        Eigen::Map<Sophus::SE3<T> const> const T_w_i(T_w_i_ptr);
        Eigen::Map<Sophus::SE3<T> const> const T_w_j(T_w_j_ptr);
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);

        //        auto T_j_i = T_w_j.inverse() * T_w_i;
        auto T_j_i = T_w_j * T_w_i.inverse();
        residual   = Sophus::se3_logd(T_i_j_.cast<T>() * T_j_i);

        residual.template head<3>() *= T(weight_translation);
        residual.template tail<3>() *= T(weight_rotation);
        //        * T(weight);
        return true;
    }

   private:
    SE3 T_i_j_;
    double weight_rotation, weight_translation;
};


struct CostDenseDepth
{
    CostDenseDepth(const Vec3& source_point, const Vec3& dest_normal, double dest_d, double weight)
        : source_point(source_point), dest_normal(dest_normal), dest_d(dest_d), weight(weight)
    {
    }

    using CostType         = CostDenseDepth;
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 1, 7, 7>;
    template <typename... Types>
    static CostFunctionType* create(const Types&... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _pose1, const T* const _pose2, T* residual_ptr) const
    {
        Eigen::Map<Sophus::SE3<T> const> const pose1(_pose1);
        Eigen::Map<Sophus::SE3<T> const> const pose2(_pose2);

        Eigen::Matrix<T, 3, 1> transformed_point = pose2 * pose1.inverse() * source_point.cast<T>();

        T res = transformed_point.dot(dest_normal.cast<T>()) + T(dest_d);

        residual_ptr[0] = res * T(weight);
        return true;
    }

   private:
    Vec3 source_point;
    Vec3 dest_normal;
    double dest_d;
    double weight;
};


}  // namespace Saiga
