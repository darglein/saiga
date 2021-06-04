/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/Range.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/kernels/BA.h"
#include "saiga/vision/kernels/Robust.h"

#include "PoseOptimizationScene.h"

#include <vector>

namespace Saiga
{
template <typename T, bool Normalized = false, Kernel::LossFunction loss_function = Kernel::LossFunction::Huber>
struct SAIGA_ALIGN_CACHE RobustPoseOptimization
{
   private:
    //#if 1
    //#    ifdef EIGEN_VECTORIZE_AVX
    //    static constexpr bool HasAvx = true;
    //#    else
    //    static constexpr bool HasAvx    = false;
    //#    endif
    //#    ifdef EIGEN_VECTORIZE_AVX512
    //    static constexpr bool HasAvx512 = true;
    //#    else
    //    static constexpr bool HasAvx512 = false;
    //#    endif

    // Only align if we can process 8 elements in one instruction
    //    static constexpr bool AlignVec4 = false;
    //        (std::is_same<T, float>::value && HasAvx) || (std::is_same<T, double>::value && HasAvx512);

    // Do triangular add if we can process 2 or less elements at once
    // -> since SSE is always enabled on x64 this happens only for doubles without avx
    //    static constexpr bool TriangularAdd = !AlignVec4 && (std::is_same<T, double>::value && !HasAvx);


   public:
    using CameraType = StereoCamera4Base<T>;
    using SE3Type    = Sophus::SE3<T>;
    using Vec2       = Eigen::Matrix<T, 2, 1>;
    using Vec3       = Eigen::Matrix<T, 3, 1>;
    using Vec4       = Eigen::Matrix<T, 4, 1>;
    using Obs        = ObsBase<T>;

    static constexpr int JParams = 6;
    //    using StereoKernel           = typename Saiga::Kernel::BAPoseStereo<T, false>;
    //    using MonoKernel             = typename Saiga::Kernel::BAPoseMono<T, false, Normalized>;
    using StereoJ = Eigen::Matrix<T, 3, 6>;
    using MonoJ   = Eigen::Matrix<T, 2, 6>;

    using JType    = Eigen::Matrix<T, JParams, JParams>;
    using BType    = Eigen::Matrix<T, JParams, 1>;
    using CompactJ = Eigen::Matrix<T, 6, 6>;
    using XType    = Eigen::Matrix<T, 6, 1>;


    RobustPoseOptimization(T thMono = 2.45, T thStereo = 2.8, T chi1Epsilon = 0.01, int maxOuterIts = 4,
                           int maxInnerIts = 10)
        : maxOuterIts(maxOuterIts), maxInnerIts(maxInnerIts)
    {
        chi1Mono         = thMono;
        chi1Stereo       = thStereo;
        deltaChi1Epsilon = chi1Epsilon;

        deltaChi2Epsilon = deltaChi1Epsilon * deltaChi1Epsilon;
        chi2Mono         = chi1Mono * chi1Mono;
        chi2Stereo       = chi1Stereo * chi1Stereo;
    }

    /**
     * Scale all thresholds by factor.
     * Usefull for example when operating in normalized image space.
     * Then you can scale the thresholds by 2/(fx+fy)
     */
    void scaleThresholds(T factor)
    {
        chi1Mono *= factor;
        chi1Stereo *= factor;
        deltaChi1Epsilon *= factor;

        deltaChi2Epsilon = deltaChi1Epsilon * deltaChi1Epsilon;
        chi2Mono         = chi1Mono * chi1Mono;
        chi2Stereo       = chi1Stereo * chi1Stereo;
    }

    int optimizePoseRobust(PoseOptimizationScene<T>& scene)
    {
        return optimizePoseRobust(scene.wps, scene.obs, scene.outlier, scene.pose, scene.K);
    }

    int optimizePoseRobust(const AlignedVector<Vec3>& wps, const AlignedVector<Obs>& obs, AlignedVector<int>& outlier,
                           SE3Type& guess, const CameraType& camera)
    {
        constexpr int threads = 1;
        locals.resize(threads);
        int N = wps.size();

        //#pragma omp parallel num_threads(threads)
        {
            for (auto outerIt : Range(0, maxOuterIts))
            {
                bool robust = outerIt < (maxOuterIts - 1);
                lastChi2sum = std::numeric_limits<T>::infinity();
                lastGuess   = guess;

                auto chi2s = chi2Stereo;
                auto chi2m = chi2Mono;
                int k      = maxOuterIts - 1 - outerIt;
                chi2s      = chi1Stereo * pow(1.2, k);
                chi2s      = chi2s * chi2s;
                chi2m      = chi1Mono * pow(1.2, k);
                chi2m      = chi2m * chi2m;

                for (auto innerIt : Range(0, maxInnerIts))
                {
                    {
                        StereoJ JrowS;
                        MonoJ JrowM;



                        auto& local = locals[OMP::getThreadNum()];
                        local.JtJ.setZero();
                        local.Jtb.setZero();
                        local.chi2    = 0;
                        local.inliers = 0;

                        //#pragma omp for
                        for (int i = 0; i < N; ++i)
                        {
                            if (outlier[i]) continue;

                            auto& o  = obs[i];
                            auto& wp = wps[i];

                            if (o.stereo())
                            {
                                auto stereo_point = o.ip(0) - camera.bf / o.depth;
                                auto [res, depth] = BundleAdjustmentStereo<T>(camera, o.ip, stereo_point, guess, wp,
                                                                           o.weight, o.weight, &JrowS, nullptr);
                                auto res_2        = res.squaredNorm();


                                // Remove outliers
                                if (outerIt > 0 && innerIt == 0)
                                {
                                    if (res_2 > chi2s || depth < 0)
                                    {
                                        outlier[i] = true;
                                        continue;
                                    }
                                }
                                T loss_weight = 1.0;
                                if (robust)
                                {
                                    auto rw     = Kernel::Loss(loss_function, chi1Stereo, res_2);
                                    res_2       = rw(0);
                                    loss_weight = rw(1);
                                }
                                local.chi2 += res_2;
                                local.JtJ += loss_weight * (JrowS.transpose() * JrowS);
                                local.Jtb -= loss_weight * JrowS.transpose() * res;
                                local.inliers++;
                            }
                            else
                            {
                                auto [res, depth] =
                                    BundleAdjustment<T>(camera, o.ip, guess, wp, o.weight, &JrowM, nullptr);
                                auto res_2 = res.squaredNorm();
                                // Remove outliers
                                if (outerIt > 0 && innerIt == 0)
                                {
                                    if (res_2 > chi2m || depth < 0)
                                    {
                                        outlier[i] = true;
                                        continue;
                                    }
                                }

                                T loss_weight = 1.0;
                                if (robust)
                                {
                                    auto rw     = Kernel::Loss(loss_function, chi1Mono, res_2);
                                    res_2       = rw(0);
                                    loss_weight = rw(1);
                                }

                                local.chi2 += res_2;
                                local.JtJ += loss_weight * (JrowM.transpose() * JrowM);
                                local.Jtb -= loss_weight * JrowM.transpose() * res;
                                local.inliers++;
                            }
                        }
                    }

                    //#pragma omp single
                    {
                        auto& JtJ = locals[0].JtJ;
                        auto& Jtb = locals[0].Jtb;

                        // sum up everything into the first local
                        for (int i = 1; i < threads; ++i)
                        {
                            locals[0].JtJ += locals[i].JtJ;
                            locals[0].Jtb += locals[i].Jtb;
                            locals[0].chi2 += locals[i].chi2;
                            locals[0].inliers += locals[i].inliers;
                        }
                        inliers = locals[0].inliers;


                        deltaChi    = lastChi2sum - locals[0].chi2;
                        lastChi2sum = locals[0].chi2;

                        if (deltaChi < 0)
                        {
                            // the error got worse :(
                            // -> discard step
                            guess = lastGuess;
                        }
                        else
                        {
                            lastGuess = guess;
                            XType x   = JtJ.ldlt().solve(Jtb);
                            guess     = Sophus::se3_expd(x) * guess;
                        }
                    }

                    if (deltaChi < 0)
                    {
                        break;
                    }

                    // early termination if the error doesn't change
                    // normalize by number of inliers
                    if (deltaChi < deltaChi2Epsilon * inliers)
                    {
                        break;
                    }
                }
            }
        }
        // We don't really need this check because the last iteration is without the robust kernel anyways
        return inliers;
    }

    struct SAIGA_ALIGN_CACHE ThreadLocalData
    {
        JType JtJ;
        BType Jtb;
        T chi2;
        int inliers;
    };

   private:
    T chi2Mono;
    T chi2Stereo;
    T chi1Mono;
    T chi1Stereo;
    T deltaChi1Epsilon;
    T deltaChi2Epsilon;
    int maxOuterIts;
    int maxInnerIts;

    // Tmp variables for OMP implementation
    AlignedVector<ThreadLocalData, SAIGA_CACHE_LINE_SIZE> locals;
    int N;
    int inliers;
    T deltaChi;
    SE3Type lastGuess;
    T lastChi2sum;

};  // namespace Saiga



}  // namespace Saiga
