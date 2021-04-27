/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "CeresArap.h"

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/ceres/CeresHelper.h"
#include "saiga/vision/ceres/local_parameterization_se3.h"

#include "Eigen/Sparse"
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
        Eigen::Map<V3> presidual(_residual);
        Eigen::Map<V3> qresidual(_residual + 3);

        V3 ij = e_ij.cast<T>();

        V3 R_eij  = pHat.so3() * ij;
        presidual = w_Reg * (pHat.translation() - qHat.translation() - R_eij);

        V3 R_eji  = qHat.so3() * -ij;
        qresidual = w_Reg * (qHat.translation() - pHat.translation() - R_eji);

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


        //        Vec3 R_eji = qHat.so3() * (-e_ij);
        //        residual += w_Reg * (qHat.translation() - pHat.translation() - R_eji);

        if (_jacobians)
        {
            if (_jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<T, 3, 7, Eigen::RowMajor>> jpose2(_jacobians[0]);
                jpose2.block<3, 3>(0, 0) = Mat3::Identity();
                jpose2.block<3, 3>(0, 3) = skew(R_eij);
                jpose2.block<3, 1>(0, 6) = Vec3::Zero();
                jpose2 *= w_Reg;
            }
            if (_jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<T, 3, 7, Eigen::RowMajor>> jpose2(_jacobians[1]);
                jpose2.block<3, 3>(0, 0) = -Mat3::Identity();
                jpose2.block<3, 3>(0, 3) = Mat3::Zero();
                jpose2.block<3, 1>(0, 6) = Vec3::Zero();
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
        Vec3 t = target.cast<T>();

        // Error
        residual = p.translation() - t;


        if (_jacobians)
        {
            if (_jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<T, 3, 7, Eigen::RowMajor>> jpose2(_jacobians[0]);
                jpose2.block<3, 3>(0, 0) = Mat3::Identity();
                //                jpose2.block<3, 4>(0, 3).setZero();
                jpose2.block<3, 3>(0, 3) = Mat3::Zero();
                jpose2.block<3, 1>(0, 6) = Vec3::Zero();
            }
        }

        return true;
    }

   protected:
    const Vec3 target;
};

#define CERES_AUTODIFF

void CeresArap::optimizeAutodiff(ArapProblem& arap, int its)
{
    // fill targets and mesh
    // ...

    auto qp = new Sophus::test::LocalParameterizationSE3();

    ceres::Problem problem;

    // Add targets
    for (int i = 0; i < (int)arap.target_indices.size(); ++i)
    {
        const auto functor = new F_Fit(arap.target_positions[i]);
        auto cost_function = new ceres::AutoDiffCostFunction<F_Fit, 3, 7>(functor);
        problem.AddResidualBlock(cost_function, nullptr, arap.vertices[arap.target_indices[i]].data());
    }

    for (auto& c2 : arap.constraints)
    {
        {
            auto c  = c2;
            int i   = c.ids.first;
            int j   = c.ids.second;
            auto& p = arap.vertices[i];
            auto& q = arap.vertices[j];

            const auto functor = new F_ARAP(sqrt(c.weight), p.translation(), q.translation());
            auto cost_function = new ceres::AutoDiffCostFunction<F_ARAP, 6, 7, 7>(functor);
            problem.AddResidualBlock(cost_function, nullptr, p.data(), q.data());
            problem.SetParameterization(q.data(), qp);
            problem.SetParameterization(p.data(), qp);
        }

        if (0)
        {
            auto c  = c2.flipped();
            int i   = c.ids.first;
            int j   = c.ids.second;
            auto& p = arap.vertices[i];
            auto& q = arap.vertices[j];

            const auto functor = new F_ARAP(sqrt(c.weight), p.translation(), q.translation());
            auto cost_function = new ceres::AutoDiffCostFunction<F_ARAP, 3, 7, 7>(functor);
            problem.AddResidualBlock(cost_function, nullptr, p.data(), q.data());
            problem.SetParameterization(q.data(), qp);
            problem.SetParameterization(p.data(), qp);
        }
    }

    //#if 0
    //    Saiga::printDebugJacobi(problem, 10);
    //#endif
    //    Saiga::printDebugSmall(problem);


    ceres::Solver::Options ceres_options;
    ceres_options.minimizer_progress_to_stdout = true;
    ceres_options.max_num_iterations           = its;



    ceres::Solver::Summary summaryTest;
    ceres::Solve(ceres_options, &problem, &summaryTest);
}



OptimizationResults CeresArap::initAndSolve()
{
    // fill targets and mesh
    // ...

    auto& arap = *_scene;

    auto qp = new Sophus::test::LocalParameterizationSE32<false>();

    ceres::Problem problem;

    // Add targets
    for (int i = 0; i < (int)arap.target_indices.size(); ++i)
    {
        auto cost_function = new FitAnalytic(arap.target_positions[i]);
        problem.AddResidualBlock(cost_function, nullptr, arap.vertices[arap.target_indices[i]].data());
    }

    for (auto& c2 : arap.constraints)
    {
        {
            auto c  = c2;
            int i   = c.ids.first;
            int j   = c.ids.second;
            auto& p = arap.vertices[i];
            auto& q = arap.vertices[j];

            const auto cost_function = new ArapAnalytic(sqrt(c.weight), p.translation(), q.translation());
            problem.AddResidualBlock(cost_function, nullptr, p.data(), q.data());
            problem.SetParameterization(q.data(), qp);
            problem.SetParameterization(p.data(), qp);
        }

        if (1)
        {
            auto c  = c2.flipped();
            int i   = c.ids.first;
            int j   = c.ids.second;
            auto& p = arap.vertices[i];
            auto& q = arap.vertices[j];

            const auto cost_function = new ArapAnalytic(sqrt(c.weight), p.translation(), q.translation());
            problem.AddResidualBlock(cost_function, nullptr, p.data(), q.data());
            problem.SetParameterization(q.data(), qp);
            problem.SetParameterization(p.data(), qp);
        }
    }



    //    ceres::Solver::Options ceres_options = make_options(optimizationOptions, false);

    ceres::Solver::Options ceres_options;
    ceres_options.minimizer_progress_to_stdout = optimizationOptions.debugOutput;
    ceres_options.linear_solver_type           = ceres::LinearSolverType::CGNR;
    ceres_options.max_num_iterations           = optimizationOptions.maxIterations;

    ceres_options.min_linear_solver_iterations = optimizationOptions.maxIterativeIterations;
    ceres_options.max_linear_solver_iterations = optimizationOptions.maxIterativeIterations;

    OptimizationResults result = ceres_solve(ceres_options, problem);
    result.name                = "Ceres ARAP";
    return result;
}


}  // namespace Saiga
