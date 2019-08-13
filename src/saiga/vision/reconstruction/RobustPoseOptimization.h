/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/Range.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/Robust.h"

#include <vector>

namespace Saiga
{
template <typename T>
struct ObsBase
{
    using Vec2 = Eigen::Matrix<T, 2, 1>;
    Vec2 ip;
    T depth  = -1;
    T weight = 1;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool stereo() const { return depth > 0; }
    template <typename G>
    ObsBase<G> cast() const
    {
        ObsBase<G> result;
        result.ip     = ip.template cast<G>();
        result.depth  = static_cast<T>(depth);
        result.weight = static_cast<T>(weight);
        return result;
    }
};


template <typename T, bool Normalized = false>
struct SAIGA_TEMPLATE SAIGA_ALIGN_CACHE RobustPoseOptimization
{
   private:
#ifdef EIGEN_VECTORIZE_AVX
    static constexpr bool HasAvx = true;
#else
    static constexpr bool HasAvx    = false;
#endif
#ifdef EIGEN_VECTORIZE_AVX512
    static constexpr bool HasAvx512 = true;
#else
    static constexpr bool HasAvx512 = false;
#endif

    // Only align if we can process 8 elements in one instruction
    static constexpr bool AlignVec4 = false;
    //        (std::is_same<T, float>::value && HasAvx) || (std::is_same<T, double>::value && HasAvx512);

    // Do triangular add if we can process 2 or less elements at once
    // -> since SSE is always enabled on x64 this happens only for doubles without avx
    static constexpr bool TriangularAdd = !AlignVec4 && (std::is_same<T, double>::value && !HasAvx);


   public:
    using CameraType = StereoCamera4Base<T>;
    using SE3Type    = Sophus::SE3<T>;
    using Vec2       = Eigen::Matrix<T, 2, 1>;
    using Vec3       = Eigen::Matrix<T, 3, 1>;
    using Vec4       = Eigen::Matrix<T, 4, 1>;
    using Obs        = ObsBase<T>;

    static constexpr int JParams = AlignVec4 ? 8 : 6;
    using StereoKernel           = typename Saiga::Kernel::BAPoseStereo<T, AlignVec4>;
    using MonoKernel             = typename Saiga::Kernel::BAPoseMono<T, AlignVec4, Normalized>;
    using StereoJ                = typename StereoKernel::JacobiType;
    using MonoJ                  = typename MonoKernel::JacobiType;
    using JType                  = Eigen::Matrix<T, JParams, JParams>;
    using BType                  = Eigen::Matrix<T, JParams, 1>;
    using CompactJ               = Eigen::Matrix<T, 6, 6>;
    using XType                  = Eigen::Matrix<T, 6, 1>;

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

    int optimizePoseRobust(const AlignedVector<Vec3>& wps, const AlignedVector<Obs>& obs, AlignedVector<int>& outlier,
                           SE3Type& guess, const CameraType& camera)
    {
        StereoJ JrowS;
        MonoJ JrowM;

        if (AlignVec4)
        {
            // clear the padded zeros
            JrowS.setZero();
            JrowM.setZero();
        }

        int N       = wps.size();
        int inliers = 0;

        for (auto outerIt : Range(0, maxOuterIts))
        {
            bool robust = outerIt < (maxOuterIts - 1);
            JType JtJ;
            BType Jtb;
            T lastChi2sum     = std::numeric_limits<T>::infinity();
            SE3Type lastGuess = guess;

            // compute current outlier threshold
            // we start a bit higher than the given
            // threshold and reduce it in each iteration
            // note: the huber threshold does not change!
            auto chi2s = chi2Stereo;
            auto chi2m = chi2Mono;
            int k      = maxOuterIts - 1 - outerIt;
            chi2s      = chi1Stereo * pow(1.2, k);
            chi2s      = chi2s * chi2s;
            chi2m      = chi1Mono * pow(1.2, k);
            chi2m      = chi2m * chi2m;

            for (auto innerIt : Range(0, maxInnerIts))
            {
                JtJ.setZero();
                Jtb.setZero();
                T chi2sum = 0;
                inliers   = 0;


                for (auto i : Range(0, N))
                {
                    if (outlier[i]) continue;

                    auto& o  = obs[i];
                    auto& wp = wps[i];

                    if (o.stereo())
                    {
                        Vec3 res;
                        StereoKernel::evaluateResidualAndJacobian(camera, guess, wp, o.ip, o.depth, res, JrowS,
                                                                  o.weight);
                        auto c2 = res.squaredNorm();
                        // Remove outliers
                        if (outerIt > 0 && innerIt == 0)
                        {
                            if (c2 > chi2s)
                            {
                                outlier[i] = true;
                                continue;
                            }
                        }
                        if (robust)
                        {
                            auto rw       = Kernel::huberWeight(chi1Stereo, c2);
                            auto sqrtLoss = sqrt(rw(1));
                            JrowS *= sqrtLoss;
                            res *= sqrtLoss;
                        }
                        chi2sum += c2;
                        JtJ += (JrowS.transpose() * JrowS);
                        Jtb += JrowS.transpose() * res;
                        inliers++;
                    }
                    else
                    {
                        Vec2 res;
                        MonoKernel::evaluateResidualAndJacobian(camera, guess, wp, o.ip, res, JrowM, o.weight);
                        auto c2 = res.squaredNorm();
#if 1
                        // Remove outliers
                        if (outerIt > 0 && innerIt == 0)
                        {
                            if (c2 > chi2m)
                            {
                                outlier[i] = true;
                                continue;
                            }
                        }
#endif

                        if (robust)
                        {
                            auto rw       = Kernel::huberWeight(chi1Mono, c2);
                            auto sqrtLoss = sqrt(rw(1));
                            JrowM *= sqrtLoss;
                            res *= sqrtLoss;
                        }

                        chi2sum += c2;
                        JtJ += (JrowM.transpose() * JrowM);
                        Jtb += JrowM.transpose() * res;
                        inliers++;
                    }
                }
                T deltaChi  = lastChi2sum - chi2sum;
                lastChi2sum = chi2sum;

#if 0
                std::cout << outerIt << " Robust: " << robust << " "
                          << " Chi2: " << chi2sum << " Delta: " << deltaChi << " Robust: " << robust
                          << " Inliers: " << inliers << "/" << N << std::endl;
#endif
                if (deltaChi < 0)
                {
                    // the error got worse :(
                    // -> discard step
                    guess = lastGuess;
                    break;
                }

#if 0
// simple LM instead of GN
                for (int k = 0; k < JtJ.rows(); ++k)
                {
                    auto& value = JtJ.diagonal()(k);
                    value       = value + 1e-3 * value;
                    value       = std::clamp(value, 1e-6, 1e32);
                }
#endif

                lastGuess = guess;


                //                std::cout << JtJ << std::endl;
                //                return 0;
                XType x;
                if constexpr (AlignVec4)
                {
                    CompactJ J = JtJ.template block<6, 6>(0, 0);
                    x          = J.ldlt().solve(Jtb.template segment<6>(0));
                }
                else
                {
                    x = JtJ.ldlt().solve(Jtb);
                }
                guess = SE3Type::exp(x) * guess;


                // early termination if the error doesn't change
                // normalize by number of inliers
                if (deltaChi < deltaChi2Epsilon * inliers)
                {
                    //                    std::cout << "inner " << innerIt << std::endl;
                    break;
                }
            }
        }
// We don't really need this check because the last iteration is without the robust kernel anyways
#if 0
        inliers = 0;
        // One more check because the last iteration changed the guess again
        for (auto i : Range(0, N))
        {
            auto& wp = wps[i];
            auto& o  = obs[i];
            if (outlier[i]) continue;
            T chi2;
            // we need to only recompute it for outliers
            if (o.depth > 0)
            {
                auto res = Saiga::Kernel::BAPoseStereo<T>::evaluateResidual(camera, guess, wp, o.ip, o.depth, o.weight);
                chi2     = res.squaredNorm();
            }
            else
            {
                auto res = Saiga::Kernel::BAPoseMono<T>::evaluateResidual(camera, guess, wp, o.ip, o.weight);
                chi2     = res.squaredNorm();
            }
            bool os = o.depth > 0 ? chi2 > chi2Stereo : chi2 > chi2Mono;
            if (!os)
            {
                inliers++;
            }
            outlier[i] = os;
        }
#else
//        inliers = 0;
//        for (auto b : outlier)
//            if (!b) inliers++;
#endif

        return inliers;
    }

    struct SAIGA_ALIGN_CACHE ThreadLocalData
    {
        JType JtJ;
        BType Jtb;
        T chi2;
        int inliers;
    };


    int optimizePoseRobustOMP(const AlignedVector<Vec3>& wps, const AlignedVector<Obs>& obs,
                              AlignedVector<int>& outlier, SE3Type& guess, const CameraType& camera, int N)
    {
#pragma omp single
        {
            //            int numThreads = 4;
            int numThreads = OMP::getMaxThreads();
            locals.resize(numThreads);
            //            N = obs.size();
            //            std::cout << "numt " << numThreads << std::endl;
        }



        //#pragma omp parallel
        {
            auto tid    = OMP::getThreadNum();
            auto& local = locals[tid];

            StereoJ JrowS;
            MonoJ JrowM;
            if constexpr (AlignVec4)
            {
                // clear the padded zeros
                JrowS.setZero();
                JrowM.setZero();
            }
            for (auto outerIt : Range(0, maxOuterIts))
            {
                bool robust = outerIt < (maxOuterIts - 1);

#pragma omp single
                {
                    lastChi2sum = std::numeric_limits<T>::infinity();
                    lastGuess   = guess;
                }

                // compute current outlier threshold
                // we start a bit higher than the given
                // threshold and reduce it in each iteration
                // note: the huber threshold does not change!
                auto chi2s = chi2Stereo;
                auto chi2m = chi2Mono;
                int k      = maxOuterIts - 1 - outerIt;
                chi2s      = chi1Stereo * pow(1.2, k);
                chi2s      = chi2s * chi2s;
                chi2m      = chi1Mono * pow(1.2, k);
                chi2m      = chi2m * chi2m;

                for (auto innerIt : Range(0, maxInnerIts))
                {
                    local.chi2 = 0;
                    local.JtJ.setZero();
                    local.Jtb.setZero();
                    local.inliers = 0;

#pragma omp single
                    {
                        JType JtJ2;
                        BType Jtb2;
                        JtJ2.setZero();
                        Jtb2.setZero();
                    }


#pragma omp for
                    for (int i = 0; i < N; ++i)
                    {
                        if (outlier[i]) continue;

                        auto& o  = obs[i];
                        auto& wp = wps[i];

                        if (o.stereo())
                        {
                            Vec3 res;
                            StereoKernel::evaluateResidualAndJacobian(camera, guess, wp, o.ip, o.depth, res, JrowS,
                                                                      o.weight);
                            auto c2 = res.squaredNorm();
                            // Remove outliers
                            if (outerIt > 0 && innerIt == 0)
                            {
                                if (c2 > chi2s)
                                {
                                    outlier[i] = true;
                                    continue;
                                }
                            }
                            if (robust)
                            {
                                auto rw       = Kernel::huberWeight(chi1Stereo, c2);
                                auto sqrtLoss = sqrt(rw(1));
                                JrowS *= sqrtLoss;
                                res *= sqrtLoss;
                            }
                            local.chi2 += c2;
                            local.JtJ += (JrowS.transpose() * JrowS);
                            local.Jtb += JrowS.transpose() * res;
                            local.inliers++;
                        }
                        else
                        {
                            Vec2 res;
                            MonoKernel::evaluateResidualAndJacobian(camera, guess, wp, o.ip, res, JrowM, o.weight);
                            auto c2 = res.squaredNorm();
#if 1
                            // Remove outliers
                            if (outerIt > 0 && innerIt == 0)
                            {
                                if (c2 > chi2m)
                                {
                                    outlier[i] = true;
                                    continue;
                                }
                            }
#endif

                            if (robust)
                            {
                                auto rw       = Kernel::huberWeight(chi1Mono, c2);
                                auto sqrtLoss = sqrt(rw(1));
                                JrowM *= sqrtLoss;
                                res *= sqrtLoss;
                            }

                            local.chi2 += c2;
                            local.JtJ += (JrowM.transpose() * JrowM);
                            local.Jtb += JrowM.transpose() * res;
                            local.inliers++;
                        }
                    }


#pragma omp single
                    {
                        T chi2sum = 0;
                        JType JtJ;
                        BType Jtb;
                        JtJ.setZero();
                        Jtb.setZero();
                        inliers = 0;
                        for (auto& l : locals)
                        {
                            chi2sum += l.chi2;
                            JtJ += l.JtJ;
                            Jtb += l.Jtb;
                            inliers += l.inliers;
                        }
                        //                        std::cout << chi2sum << std::endl;

                        deltaChi    = lastChi2sum - chi2sum;
                        lastChi2sum = chi2sum;
                        if (deltaChi < 0)
                        {
                            // the error got worse :(
                            // -> discard step
                            guess = lastGuess;
                        }
                        else
                        {
                            lastGuess = guess;
                            XType x;
                            if constexpr (AlignVec4)
                            {
                                CompactJ J = JtJ.template block<6, 6>(0, 0);
                                x          = J.ldlt().solve(Jtb.template segment<6>(0));
                            }
                            else
                            {
                                x = JtJ.ldlt().solve(Jtb);
                            }
                            guess = SE3Type::exp(x) * guess;
                        }
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
        return inliers;
    }

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
