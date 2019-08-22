#pragma once

#include <sophus/sim3.hpp>

#include <ceres/local_parameterization.h>

namespace Sophus
{
namespace test
{
template <bool LEFT_MULT = true>
class LocalParameterizationSim3 : public ceres::LocalParameterization
{
   public:
    virtual ~LocalParameterizationSE3() {}

    // SE3 plus operation for Ceres
    //
    //  T * exp(x)
    //
    virtual bool Plus(double const* T_raw, double const* delta_raw, double* T_plus_delta_raw) const
    {
        Eigen::Map<Sim3d const> const T(T_raw);
        Eigen::Map<Vector7d const> const delta(delta_raw);
        Eigen::Map<Sim3d> T_plus_delta(T_plus_delta_raw);
        if (LEFT_MULT)
        {
            T_plus_delta = Sim3d::exp(delta) * T;
        }
        else
        {
            T_plus_delta = T * Sim3d::exp(delta);
        }
        return true;
    }

    // Jacobian of SE3 plus operation for Ceres
    //
    // Dx T * exp(x)  with  x=0
    //
    virtual bool ComputeJacobian(double const* T_raw, double* jacobian_raw) const
    {
        Eigen::Map<Sim3d const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian(jacobian_raw);

        if (LEFT_MULT)
        {
            jacobian = -T.Dx_this_mul_exp_x_at_0();
        }
        else
        {
            jacobian = T.Dx_this_mul_exp_x_at_0();
        }
        return true;
    }

    virtual int GlobalSize() const { return Sim3d::num_parameters; }

    virtual int LocalSize() const { return Sim3d::DoF; }
};


}  // namespace test
}  // namespace Sophus


