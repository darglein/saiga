/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/kernels/PGO.h"

#include "ceres/autodiff_cost_function.h"

namespace Saiga
{
struct CostPGO
{
    CostPGO(const SE3& T_i_j, double weight = 1) : T_i_j_(T_i_j), weight(weight) {}

    using CostType         = CostPGO;
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, SE3::DoF, 7, 7>;
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
        Eigen::Map<Eigen::Matrix<T, SE3::DoF, 1>> residual(residual_ptr);

        auto T_j_i = T_w_j.inverse() * T_w_i;
        residual   = Sophus::se3_logd(T_i_j_.cast<T>() * T_j_i) * T(weight);
        return true;
    }

   private:
    SE3 T_i_j_;
    double weight;
};


struct CostPGODSim3
{
    CostPGODSim3(const DSim3& T_i_j, double weight = 1) : T_i_j_(T_i_j), weight(weight) {}

    using CostType         = CostPGODSim3;
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 7, 8, 8>;
    template <typename... Types>
    static CostFunctionType* create(const Types&... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const T_w_i_ptr, const T* const T_w_j_ptr, T* residual_ptr) const
    {
        Sophus::DSim3<T> T_w_i = Sophus::PointerToDSim3(T_w_i_ptr);
        Sophus::DSim3<T> T_w_j = Sophus::PointerToDSim3(T_w_j_ptr);

        //        std::cout << T_w_i << std::endl;
        //        std::cout << T_w_j << std::endl;
        Eigen::Map<Eigen::Matrix<T, 7, 1>> residual(residual_ptr);

        auto T_j_i = T_w_j.inverse() * T_w_i;
        residual   = Sophus::dsim3_logd(T_i_j_.cast<T>() * T_j_i) * T(weight);
        return true;
    }

   private:
    DSim3 T_i_j_;
    double weight;
};


#if 0
class CostPGOAnalytic : public ceres::SizedCostFunction<7, 7, 7>
{
   public:
    using PGOTransformation = SE3;

    static constexpr int DOF = PGOTransformation::DoF;
    using T                  = double;

    using Kernel = Saiga::Kernel::PGO<double>;

    CostPGOAnalytic(const PGOTransformation& invMeassurement, double weight = 1)
        : _inverseMeasurement(invMeassurement), weight(weight)
    {
    }

    virtual ~CostPGOAnalytic() {}

    virtual bool Evaluate(double const* const* _parameters, double* _residuals, double** _jacobians) const
    {
        Eigen::Map<Sophus::SE3<T> const> const from(_parameters[0]);
        Eigen::Map<Sophus::SE3<T> const> const to(_parameters[1]);
        Eigen::Map<Eigen::Matrix<T, DOF, 1>> residual(_residuals);



        if (!_jacobians)
        {
            // only compute residuals
            Kernel::ResidualType res;
            Kernel::evaluateResidual(from, to, _inverseMeasurement, res, weight);
            residual = res;
        }
        else
        {
            // compute both
            Kernel::PoseJacobiType JrowFrom, JrowTo;
            Kernel::ResidualType res;
            Kernel::evaluateResidualAndJacobian(from, to, _inverseMeasurement, res, JrowFrom, JrowTo, weight);

            residual = res;



            if (_jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<T, DOF, 7, Eigen::RowMajor>> jpose2(_jacobians[0]);
                jpose2.setZero();
                jpose2.block<DOF, DOF>(0, 0) = JrowFrom;
            }
            if (_jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<T, DOF, 7, Eigen::RowMajor>> jpose2(_jacobians[1]);
                jpose2.setZero();
                jpose2.block<DOF, DOF>(0, 0) = JrowTo;
            }
        }


        return true;
    }

   private:
    PGOTransformation _inverseMeasurement;
    double weight;
};
#endif

}  // namespace Saiga
