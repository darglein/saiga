/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Trajectory.h"

#include "saiga/core/util/Range.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/icp/ICPAlign.h"


#ifdef SAIGA_USE_CERES
#    include "saiga/vision/ceres/CeresHelper.h"
#    include "saiga/vision/ceres/local_parameterization_se3.h"
#    include "saiga/vision/ceres/local_parameterization_sim3.h"

#    include "ceres/autodiff_cost_function.h"
#endif

namespace Saiga
{
namespace Trajectory
{
std::ostream& operator<<(std::ostream& strm, const Scene& scene)
{
    strm << "[Trajectory] N = " << scene.vertices.size() << " Scale = " << scene.scale
         << " T = " << scene.transformation << " Chi2 = " << scene.chi2();
    return strm;
}

void Scene::InitialAlignment()
{
    // fit trajectories with icp
    AlignedVector<ICP::Correspondence> corrs;
    for (auto& v : vertices)
    {
        ICP::Correspondence c;
        // c.srcPoint = v.estimate.translation();
        c.srcPoint = (v.estimate * extrinsics).translation();
        c.refPoint = v.ground_truth.translation();
        corrs.push_back(c);
    }

    SE3 relSe3 = ICP::pointToPointDirect(corrs, optimize_scale ? &scale : nullptr);


    transformation = relSe3;

    if (scale < 0.0001 || scale > 1000 || !std::isfinite(scale))
    {
        std::cout << "Trajectory::align invalid scale: " << scale << std::endl;
    }
}

#ifdef SAIGA_USE_CERES



struct CostAlign
{
    // Helper function to simplify the "add residual" part for creating ceres problems
    using CostType = CostAlign;
    // Note: The first number is the number of residuals
    //       The following number sthe size of the residual blocks (without local parametrization)
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 3, 7, 7, 1>;
    template <typename... Types>
    static CostFunctionType* create(Types... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _extrinsics, const T* const _transformation, const T* const _scale,
                    T* _residuals) const
    {
        Eigen::Map<Sophus::SE3<T> const> const extrinsics(_extrinsics);
        Eigen::Map<Sophus::SE3<T> const> const transformation(_transformation);
        const T& scale = *_scale;

        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(_residuals);


        Sophus::SE3<T> e = obs.estimate.cast<T>();
        e.translation() *= scale;
        e = e * extrinsics;
        e = transformation * e;

        residual = e.translation() - obs.ground_truth.translation().cast<T>();

        return true;
    }

    CostAlign(const Observation& obs) : obs(obs) {}

    Observation obs;
};



void Scene::OptimizeCeres()
{
    ceres::Problem::Options problemOptions;
    ceres::Problem problem(problemOptions);

    auto se3_parameterization = new Sophus::test::LocalParameterizationSE3;

    problem.AddParameterBlock(extrinsics.data(), 7, se3_parameterization);
    problem.AddParameterBlock(transformation.data(), 7, se3_parameterization);
    problem.AddParameterBlock(&scale, 1);

    if (!optimize_scale)
    {
        problem.SetParameterBlockConstant(&scale);
    }

    if (!optimize_extrinsics)
    {
        problem.SetParameterBlockConstant(extrinsics.data());
    }

    for (auto& v : vertices)
    {
        auto cost = CostAlign::create(v);
        problem.AddResidualBlock(cost, nullptr, extrinsics.data(), transformation.data(), &scale);
    }

    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations           = 50;
    ceres_options.minimizer_progress_to_stdout = false;
    ceres_options.function_tolerance           = 1e-30;
    ceres_options.gradient_tolerance           = 1e-30;
    ceres_options.parameter_tolerance          = 1e-30;
    ceres_solve(ceres_options, problem);
}
#endif

std::pair<SE3, double> align(ArrayView<std::pair<int, SE3>> A, ArrayView<std::pair<int, SE3>> B, bool computeScale)
{
    SAIGA_ASSERT(A.size() == B.size());
    if (A.size() == 0)
    {
        return {{}, 0};
    }
    int N = A.size();

    auto compFirst = [](const std::pair<int, SE3>& a, const std::pair<int, SE3>& b) { return a.first < b.first; };
    std::sort(A.begin(), A.end(), compFirst);
    std::sort(B.begin(), B.end(), compFirst);

    // transform both trajectories so that the first kf is at the origin
    //    SE3 pinv1 = A.front().second.inverse();
    //    SE3 pinv2 = B.front().second.inverse();
    //    for (auto& m : A) m.second = pinv1 * m.second;
    //    for (auto& m : B) m.second = pinv2 * m.second;


    // fit trajectories with icp
    AlignedVector<ICP::Correspondence> corrs;
    for (int i = 0; i < (int)A.size(); ++i)
    {
        ICP::Correspondence c;
        c.srcPoint = A[i].second.translation();
        c.refPoint = B[i].second.translation();
        corrs.push_back(c);
    }

    double scale = 1;
    SE3 relSe3   = ICP::pointToPointDirect(corrs, computeScale ? &scale : nullptr);


    if (scale < 0.0001 || scale > 1000 || !std::isfinite(scale))
    {
        std::cout << "Trajectory::align invalid scale." << std::endl;
        return {{}, 0};
    }

    DSim3 rel(relSe3, scale);



    // Apply transformation to the src trajectory (A)
    double error = 0;
    for (auto i : Range(0, N))
    {
        auto& c = corrs[i];
        error += c.residualPointToPoint();
        A[i].second = (rel * DSim3(A[i].second, 1.0)).se3();
    }

    return {relSe3, scale};
}

std::vector<double> rpe(ArrayView<const std::pair<int, SE3>> A, ArrayView<const std::pair<int, SE3>> B, int difference)
{
    SAIGA_ASSERT(A.size() == B.size());
    int N = A.size();
    std::vector<double> rpe;
    if (A.empty()) return rpe;


    if (N < difference) return rpe;

    for (auto i : Range(difference, N))
    {
        auto [a_id, a_se] = A[i];
        auto [b_id, b_se] = B[i];
        SAIGA_ASSERT(a_id == b_id);


        auto [a_id_prev, a_se_prev] = A[i - difference];
        auto b_se_prev              = B[i - difference].second;

        auto a_rel = a_se.inverse() * a_se_prev;
        auto b_rel = b_se.inverse() * b_se_prev;

        auto et              = translationalError(a_rel, b_rel);
        int numFramesBetween = a_id - a_id_prev;
        SAIGA_ASSERT(numFramesBetween > 0);
        et = et / double(numFramesBetween);
        rpe.push_back(et);
    }
    return rpe;
}

std::vector<double> ate(ArrayView<const std::pair<int, SE3>> A, ArrayView<const std::pair<int, SE3>> B)
{
    SAIGA_ASSERT(A.size() == B.size());
    int N = A.size();

    std::vector<double> ate;
    for (auto i : Range(0, N))
    {
        auto et = translationalError(A[i].second, B[i].second);
        ate.push_back(et);
    }
    return ate;
}

std::vector<double> are(ArrayView<const std::pair<int, SE3>> A, ArrayView<const std::pair<int, SE3>> B)
{
    SAIGA_ASSERT(A.size() == B.size());
    int N = A.size();

    std::vector<double> ate;
    for (auto i : Range(0, N))
    {
        auto et = degrees(rotationalError(A[i].second, B[i].second));
        ate.push_back(et);
    }
    return ate;
}



}  // namespace Trajectory
}  // namespace Saiga
