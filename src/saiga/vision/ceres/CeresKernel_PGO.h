/**
 * Copyright (c) 2017 Darius Rückert
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
    CostPGO(const PGOTransformation& invMeassurement, double weight = 1)
        : _inverseMeasurement(invMeassurement), weight(weight)
    {
    }

    using CostType         = CostPGO;
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, PGOTransformation::DoF, 7, 7>;
    template <typename... Types>
    static CostFunctionType* create(const Types&... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _from, const T* const _to, T* _residual) const
    {
        Eigen::Map<Sophus::Sim3<T> const> const from(_from);
        Eigen::Map<Sophus::Sim3<T> const> const to(_to);
        Eigen::Map<Eigen::Matrix<T, PGOTransformation::DoF, 1>> residual(_residual);

        Sophus::Sim3<T> inverseMeasurement = _inverseMeasurement.cast<T>();
#ifdef LSD_REL
        auto error_ = from.inverse() * to * inverseMeasurement;
#else
        auto error_ = inverseMeasurement.inverse() * from * to.inverse();
#endif

        residual = error_.log() * weight;


        return true;
    }

   private:
    PGOTransformation _inverseMeasurement;
    double weight;
};


class CostPGOAnalytic : public ceres::SizedCostFunction<PGOTransformation::DoF, 7, 7>
{
   public:
    static constexpr int DOF = PGOTransformation::DoF;
    using T                  = double;

    using Kernel = Saiga::Kernel::PGO<PGOTransformation>;

    CostPGOAnalytic(const PGOTransformation& invMeassurement, double weight = 1)
        : _inverseMeasurement(invMeassurement), weight(weight)
    {
    }

    virtual ~CostPGOAnalytic() {}

    virtual bool Evaluate(double const* const* _parameters, double* _residuals, double** _jacobians) const
    {
        Eigen::Map<Sophus::Sim3<T> const> const from(_parameters[0]);
        Eigen::Map<Sophus::Sim3<T> const> const to(_parameters[1]);
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

}  // namespace Saiga
