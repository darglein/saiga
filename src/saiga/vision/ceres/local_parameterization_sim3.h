#pragma once

#include <sophus/sim3.hpp>

#include <ceres/local_parameterization.h>

namespace Sophus
{
namespace test
{
// use for analytical diff with lie algebra
// template <bool CONSTANT = false>
class LocalParameterizationSim32 : public ceres::LocalParameterization
{
   public:
    virtual ~LocalParameterizationSim32() {}

    // SE3 plus operation for Ceres
    //
    //  T * exp(x)
    //
    virtual bool Plus(double const* T_raw, double const* delta_raw, double* T_plus_delta_raw) const
    {
        //        if (CONSTANT) return true;


        Eigen::Map<Sim3d const> const T(T_raw);
        Eigen::Map<Vector7d const> const delta(delta_raw);


        Eigen::Map<Sim3d> T_plus_delta(T_plus_delta_raw);

        Vector7d delta2 = delta;

        if (fixScale) delta2[6] = 0;

        //        std::cout << delta.transpose() << std::endl;
        //        std::cout << Sim3d::exp(delta).rxso3().quaternion().coeffs().transpose() << std::endl;


        T_plus_delta = Sim3d::exp(delta2) * T;


        return true;
    }

    // Jacobian of SE3 plus operation for Ceres
    //
    // Dx T * exp(x)  with  x=0
    //
    virtual bool ComputeJacobian(double const* T_raw, double* jacobian_raw) const
    {
        Eigen::Map<Eigen::Matrix<double, 7, 7, Eigen::RowMajor>> jacobian_r(jacobian_raw);
        Eigen::Map<const Eigen::Matrix<double, 7, 7, Eigen::RowMajor>> T_r(T_raw);

        jacobian_r.setZero();
        jacobian_r.block<7, 7>(0, 0).setIdentity();
        //        jacobian_r(6, 6) = 0;

        //        jacobian_r = T_r;

        return true;
    }


    bool fixScale = false;

    virtual int GlobalSize() const { return Sim3d::num_parameters; }
    virtual int LocalSize() const { return Sim3d::DoF; }
};

}  // namespace test
}  // namespace Sophus
