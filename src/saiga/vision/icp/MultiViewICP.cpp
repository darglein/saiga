/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "MultiViewICP.h"

#include "MultiViewICPAlign.h"
//#define WITH_CERES

#ifdef WITH_CERES
#    include "saiga/vision/CeresKernelHelper.h"
#    include "saiga/vision/local_parameterization_se3.h"

#    include "ceres/ceres.h"
#    include "ceres/problem.h"
#    include "ceres/rotation.h"
#    include "ceres/solver.h"
#endif
#include "saiga/core/time/timer.h"
#include "saiga/core/util/assert.h"

namespace Saiga
{
using Depthmap::DepthMap;

namespace ICP
{
#ifdef WITH_CERES
struct CostICPPlane
{
    // Helper function to simplify the "add residual" part for creating ceres problems
    using CostType = CostICPPlane;
    // Note: The first number is the number of residuals
    //       The following number sthe size of the residual blocks (without local parametrization)
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 1, 7, 7>;
    template <typename... Types>
    static CostFunctionType* create(Types... args)
    {
        return new CostFunctionType(new CostType(args...));
    }


    template <typename T>
    bool operator()(const T* const _ref, const T* const _src, T* _residuals) const
    {
        Eigen::Map<Sophus::SE3<T> const> const ref(_ref);
        Eigen::Map<Sophus::SE3<T> const> const src(_src);

        using Vec3T = Eigen::Matrix<T, 3, 1>;
        Vec3T rp    = convertVec<T>(_rp);
        Vec3T rn    = convertVec<T>(_rn);
        Vec3T sp    = convertVec<T>(_sp);

        rp = ref * rp;
        rn = ref.so3() * rn;
        sp = src * sp;

        Vec3T di = rp - sp;
        T res    = rn.dot(di);

        _residuals[0] = res * T(weight);

        return true;
    }

    CostICPPlane(Vec3 rp, Vec3 rn, Vec3 sp, double weight) : _rp(rp), _rn(rn), _sp(sp), weight(weight) {}

    // Ref
    Vec3 _rp;
    Vec3 _rn;
    // Src
    Vec3 _sp;
    double weight;
};

SE3 pointToPlaneCeres(const std::vector<Correspondence>& corrs, const SE3& _ref, const SE3& _src, int innerIterations)
{
    SAIGA_ASSERT(corrs.size() >= 6);
    Sophus::test::LocalParameterizationSE3* camera_parameterization = new Sophus::test::LocalParameterizationSE3;
    ceres::Problem problem;


    SE3 ref = _ref;
    SE3 src = _src;

    for (size_t i = 0; i < corrs.size(); ++i)
    {
        auto c = corrs[i];

        auto cost_function = CostICPPlane::create(c.refPoint, c.refNormal, c.srcPoint, c.weight);
        problem.AddResidualBlock(cost_function, nullptr, ref.data(), src.data());
    }

    problem.SetParameterization(ref.data(), camera_parameterization);
    problem.SetParameterization(src.data(), camera_parameterization);

    problem.SetParameterBlockConstant(ref.data());



    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations           = innerIterations;
    ceres::Solver::Summary summaryTest;
    ceres::Solve(options, &problem, &summaryTest);

    return src;
}


SE3 alignDepthMapsCeres(DepthMap referenceDepthMap, DepthMap sourceDepthMap, SE3 refPose, SE3 srcPose,
                        Intrinsics4 camera, int iterations)
{
    DepthMapExtended ref(referenceDepthMap, camera, refPose);
    DepthMapExtended src(sourceDepthMap, camera, srcPose);


    std::vector<Saiga::ICP::Correspondence> corrs;

    for (int k = 0; k < iterations; ++k)
    {
        corrs = Saiga::ICP::projectiveCorrespondences(ref, src);
        //        src.pose = Saiga::ICP::pointToPlaneCeres(corrs, ref.pose, src.pose, 1);
        src.pose = Saiga::ICP::pointToPlane(corrs, ref.pose, src.pose, 1);
    }
    return src.pose;
}

#endif

void multiViewICPSimple(const std::vector<Depthmap::DepthMap>& depthMaps, AlignedVector<SE3>& guesses,
                        IntrinsicsPinholed camera, int iterations, ProjectiveCorrespondencesParams params)
{
    //    SAIGA_BLOCK_TIMER;
    // Compute all relative to the first frame
    auto ref  = depthMaps.front();
    auto refT = guesses.front();  // W <- A


    for (size_t i = 1; i < depthMaps.size(); ++i)
    {
        auto src  = depthMaps[i];
        auto srcT = guesses[i];  // W <- B

        srcT = alignDepthMaps(ref, src, refT, srcT, camera, iterations, params);
        //        srcT = alignDepthMapsCeres(ref, src, refT, srcT, camera, iterations);

        guesses[i] = srcT;  // W <- B
    }
}



void multiViewICP(const std::vector<Depthmap::DepthMap>& depthMaps, AlignedVector<SE3>& guesses,
                  IntrinsicsPinholed camera,
                  int iterations, ProjectiveCorrespondencesParams params)
{
    // initial alignment with reduced resolution
    auto initParams = params;
    initParams.stride *= 2;
    multiViewICPSimple(depthMaps, guesses, camera, iterations, params);

    for (int k = 0; k < iterations; ++k)
    {
        size_t N = depthMaps.size();

        std::vector<DepthMapExtended> dmes;

        for (size_t i = 0; i < depthMaps.size(); ++i)
        {
            dmes.emplace_back(depthMaps[i], camera, guesses[i]);
        }


        std::vector<std::pair<size_t, size_t>> pairs;
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = i + 1; j < N; ++j)
            {
                pairs.emplace_back(i, j);
            }
        }

        // find all pairwise correspondences
        std::vector<AlignedVector<Correspondence>> corrs;
        for (auto p : pairs)
        {
            auto c = projectiveCorrespondences(dmes[p.first], dmes[p.second], params);
            corrs.push_back(c);
        }
        multiViewICPAlign(N, pairs, corrs, guesses, 2);
    }
}



}  // namespace ICP
}  // namespace Saiga
