#pragma once


#include "saiga/core/sophus/Sophus.h"

#include <ceres/autodiff_local_parameterization.h>
#include <ceres/local_parameterization.h>

namespace Saiga
{
namespace test
{
template <bool FIX_SCALE>
struct Sim3Plus
{
    template <typename T>
    bool operator()(const T* _x, const T* _delta, T* _x_plus_delta) const
    {
        static_assert(!FIX_SCALE, "sdf");
        using Vec7 = Eigen::Matrix<T, 7, 1>;

        Eigen::Map<Sophus::Sim3<T> const> const x(_x);
        Eigen::Map<Vec7 const> const delta(_delta);
        Eigen::Map<Sophus::Sim3<T>> x_plus_delta(_x_plus_delta);

        // make a copy so we can set the scale to zero
        Vec7 delta2 = delta;
        if (FIX_SCALE) delta2[6] = T(0);



        //        x_plus_delta = x * Sophus::Sim3<T>::exp(-delta2);
        x_plus_delta = Sophus::Sim3<T>::exp(-delta2) * x;

        return true;
    }
};

template <bool FIX_SCALE>
using LocalParameterizationSim3 = ceres::AutoDiffLocalParameterization<Sim3Plus<FIX_SCALE>, 7, 7>;

/**
 * The local Parameterization of a Sim3 with a Identity jacobian.
 * This means, that the jacobians of the cost functions must also be derived in respect to PLUS.
 *
 * Only use if you know what you're doing!
 * Otherwise use the typedef above this comment.
 */
template <bool FIX_SCALE>
class LocalParameterizationSim3_IdentityJ : public ceres::LocalParameterization
{
   public:
    virtual ~LocalParameterizationSim3_IdentityJ() {}

    virtual bool Plus(double const* T_raw, double const* delta_raw, double* T_plus_delta_raw) const
    {
        Sim3Plus<FIX_SCALE> op;
        return op(T_raw, delta_raw, T_plus_delta_raw);
    }

    virtual bool ComputeJacobian(double const* T_raw, double* jacobian_raw) const
    {
        Eigen::Map<Eigen::Matrix<double, 7, 7, Eigen::RowMajor>> jacobian_r(jacobian_raw);
        Eigen::Map<const Eigen::Matrix<double, 7, 7, Eigen::RowMajor>> T_r(T_raw);
        jacobian_r.setZero();
        jacobian_r.block<7, 7>(0, 0).setIdentity();
        return true;
    }

    virtual int GlobalSize() const { return Sophus::Sim3d::num_parameters; }
    virtual int LocalSize() const { return Sophus::Sim3d::DoF; }
};

}  // namespace test
}  // namespace Saiga
