/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "CeresArap.h"

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/ceres/local_parameterization_se3.h"

#include "ceres/ceres.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"
#include "ceres/solver.h"

#include "ceres/local_parameterization.h"
namespace Saiga
{
class F_ARAP
{
   public:
    F_ARAP(double w_Reg, const Vec3& p, const Vec3& q) : e_ij(p - q), w_Reg(w_Reg) {}

    template <typename T>
    bool operator()(const T* const _pHat, const T* const _qHat, T* _residual) const
    {
        using V3  = Eigen::Matrix<T, 3, 1>;
        using SE3 = Sophus::SE3<T>;

        Eigen::Map<const SE3> pHat(_pHat);
        Eigen::Map<const SE3> qHat(_qHat);
        Eigen::Map<V3> residual(_residual);

        V3 ij    = e_ij.cast<T>();
        V3 R_eij = pHat.so3() * ij;

        residual = w_Reg * (pHat.translation() - qHat.translation() - R_eij);
        return true;
    }

   protected:
    const Vec3 e_ij;
    const double w_Reg;
};

class ArapAnalytic : public ceres::SizedCostFunction<3, 7, 7>
{
   public:
    using T = double;


    ArapAnalytic(double w_Reg, const Vec3& p, const Vec3& q) : e_ij(p - q), w_Reg(w_Reg) {}


    virtual bool Evaluate(double const* const* _parameters, double* _residuals, double** _jacobians) const
    {
        Eigen::Map<Sophus::SE3<T> const> const pHat(_parameters[0]);
        Eigen::Map<Sophus::SE3<T> const> const qHat(_parameters[1]);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(_residuals);


        Vec3 R_eij = pHat.so3() * e_ij;
        residual   = w_Reg * (pHat.translation() - qHat.translation() - R_eij);


        if (_jacobians)
        {
            if (_jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<T, 6, 7, Eigen::RowMajor>> jpose2(_jacobians[0]);
                jpose2.setZero();

                jpose2(0, 0) = -1;
                jpose2(1, 1) = -1;
                jpose2(2, 2) = -1;

                jpose2.block(0, 3, 3, 3) = skew(R_eij);

                jpose2 *= -1;

                jpose2 *= w_Reg;
            }
            if (_jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<T, 6, 7, Eigen::RowMajor>> jpose2(_jacobians[1]);
                jpose2.setZero();

                jpose2(0, 0) = 1;
                jpose2(1, 1) = 1;
                jpose2(2, 2) = 1;

                jpose2 *= -1;

                jpose2 *= w_Reg;
            }
        }

        return true;
    }

   protected:
    const Vec3 e_ij;
    const double w_Reg;
};

class F_Fit
{
   public:
    F_Fit(const Vec3& target) : target(target) {}

    template <typename T>
    bool operator()(const T* const _p, T* _residual) const
    {
        using V3  = Eigen::Matrix<T, 3, 1>;
        using SE3 = Sophus::SE3<T>;

        Eigen::Map<const SE3> p(_p);
        Eigen::Map<V3> residual(_residual);
        V3 t = target.cast<T>();

        residual = p.translation() - t;
        return true;
    }

   protected:
    const Vec3 target;
};



class FitAnalytic : public ceres::SizedCostFunction<3, 7>
{
   public:
    using T = double;


    FitAnalytic(const Vec3& target) : target(target) {}


    virtual bool Evaluate(double const* const* _parameters, double* _residuals, double** _jacobians) const
    {
        Eigen::Map<Sophus::SE3<T> const> const p(_parameters[0]);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(_residuals);

        Vec3 t   = target.cast<T>();
        residual = p.translation() - t;


        if (_jacobians)
        {
            if (_jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<T, 6, 7, Eigen::RowMajor>> jpose2(_jacobians[0]);
                jpose2.setZero();

                jpose2(0, 0) = 1;
                jpose2(1, 1) = 1;
                jpose2(2, 2) = 1;
            }
        }

        return true;
    }

   protected:
    const Vec3 target;
};

#define CERES_AUTODIFF

void CeresArap::optimize(ArapProblem& arap, int its)
{
// fill targets and mesh
// ...

//    auto qp = new ceres::EigenQuaternionParameterization();
#ifdef CERES_AUTODIFF
    auto qp = new Sophus::test::LocalParameterizationSE3();
#else
    auto qp = new Sophus::test::LocalParameterizationSE32();
#endif

    ceres::Problem problem;

    // Add targets
    for (int i = 0; i < arap.target_indices.size(); ++i)
    {
#ifdef CERES_AUTODIFF
        const auto functor = new F_Fit(arap.target_positions[i]);
        auto cost_function = new ceres::AutoDiffCostFunction<F_Fit, 3, 7>(functor);
        problem.AddResidualBlock(cost_function, nullptr, arap.vertices[arap.target_indices[i]].data());
#else
        auto cost_function = new FitAnalytic(arap.target_positions[i]);
        problem.AddResidualBlock(cost_function, nullptr, arap.vertices[arap.target_indices[i]].data());
#endif
    }

    for (auto& c : arap.constraints)
    {
        auto& p = arap.vertices[c.i];
        auto& q = arap.vertices[c.j];

#ifdef CERES_AUTODIFF
        const auto functor = new F_ARAP(sqrt(c.weight), p.translation(), q.translation());
        auto cost_function = new ceres::AutoDiffCostFunction<F_ARAP, 3, 7, 7>(functor);
        problem.AddResidualBlock(cost_function, nullptr, p.data(), q.data());
        problem.SetParameterization(q.data(), qp);
        problem.SetParameterization(p.data(), qp);
#else
        const auto cost_function = new ArapAnalytic(sqrt(c.weight), p.translation(), q.translation());
        problem.AddResidualBlock(cost_function, nullptr, p.data(), q.data());
        problem.SetParameterization(q.data(), qp);
        problem.SetParameterization(p.data(), qp);
#endif
    }



    ceres::Solver::Options ceres_options;
    ceres_options.minimizer_progress_to_stdout = true;
    ceres_options.max_num_iterations           = its;

    ceres::Solver::Summary summaryTest;
    ceres::Solve(ceres_options, &problem, &summaryTest);
}

}  // namespace Saiga
