/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/math/all.h"


namespace Saiga{

template <typename T, int num_params, int num_residuals, typename Functor>
Eigen::Matrix<T, num_residuals, 1> EvaluateNumeric(Functor f, const Eigen::Matrix<T, num_params, 1>& params,
                                                   Matrix<double, num_residuals, num_params>* jacobian = nullptr, double eps = 1e-4)
{
    Eigen::Matrix<T, num_residuals, 1> residual = f(params);

    if (jacobian)
    {
        double eps_scale = 1.0 / (2.0 * eps);
        for (int i = 0; i < num_params; ++i)
        {
            auto param_copy = params;
            param_copy(i) -= eps;
            auto low = f(param_copy);

            param_copy = params;
            param_copy(i) += eps;
            auto high = f(param_copy);

            auto diff        = (high - low) * eps_scale;
            jacobian->col(i) = diff.transpose();
        }
    }

    return residual;
}

}
