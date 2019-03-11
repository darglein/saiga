﻿/**
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
class CostPGOAnalytic : public ceres::SizedCostFunction<6, 7, 7>
{
   public:
    using T = double;

    using Kernel = Saiga::Kernel::PGO<double>;

    CostPGOAnalytic(const SE3& invMeassurement, double weight = 1)
        : _inverseMeasurement(invMeassurement), weight(weight)
    {
    }

    virtual ~CostPGOAnalytic() {}

    virtual bool Evaluate(double const* const* _parameters, double* _residuals, double** _jacobians) const
    {
        Eigen::Map<Sophus::SE3<T> const> const from(_parameters[0]);
        Eigen::Map<Sophus::SE3<T> const> const to(_parameters[1]);
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(_residuals);



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
                Eigen::Map<Eigen::Matrix<T, 6, 7, Eigen::RowMajor>> jpose2(_jacobians[0]);
                jpose2.setZero();
                jpose2.block<6, 6>(0, 0) = JrowFrom;
            }
            if (_jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<T, 6, 7, Eigen::RowMajor>> jpose2(_jacobians[1]);
                jpose2.setZero();
                jpose2.block<6, 6>(0, 0) = JrowTo;
            }
        }

        return true;
    }

   private:
    SE3 _inverseMeasurement;
    double weight;
};

}  // namespace Saiga