/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/Range.h"
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

    template <typename G>
    ObsBase<G> cast()
    {
        ObsBase<G> result;
        result.ip     = ip.template cast<G>();
        result.depth  = static_cast<T>(depth);
        result.weight = static_cast<T>(weight);
        return result;
    }
};

using Obs = ObsBase<double>;

template <typename T, bool TriangularAdd>
struct SAIGA_VISION_API SAIGA_ALIGN_CACHE RobustPoseOptimization
{
    using CameraType = StereoCamera4Base<T>;
    using SE3Type    = Sophus::SE3<T>;
    using Vec2       = Eigen::Matrix<T, 2, 1>;
    using Vec3       = Eigen::Matrix<T, 3, 1>;
    using Vec4       = Eigen::Matrix<T, 4, 1>;
    using Obs        = ObsBase<T>;


    T chi2Mono        = 5.991;
    T chi2Stereo      = 7.815;
    T deltaMono       = sqrt(5.991);
    T deltaStereo     = sqrt(7.815);
    T deltaChiEpsilon = 1e-4;
    int maxOuterIts   = 4;
    int maxInnerIts   = 10;

   private:
    std::vector<T> chi2s;

   public:
    int optimizePoseRobust(const AlignedVector<Vec3>& wps, const AlignedVector<Obs>& obs, AlignedVector<bool>& outlier,
                           SE3Type& guess, const CameraType& camera)
    {
        int N            = wps.size();
        int inliers      = 0;
        T innerThreshold = deltaChiEpsilon;
        chi2s.resize(N);
        SE3Type pose;
        for (auto outerIt : Range(0, maxOuterIts))
        {
            //            cout << "outer " << outerIt << endl;
            bool robust = outerIt <= 2;
            Eigen::Matrix<T, 6, 6> JtJ;
            Eigen::Matrix<T, 6, 1> Jtb;
            pose          = guess;
            T lastChi2sum = 1e50;
            //            SE3Type lastGuess = guess;

            for (auto innerIt : Range(0, maxInnerIts))
            {
                //                cout << "inner " << innerIt << endl;
                JtJ.setZero();
                Jtb.setZero();
                T chi2sum = 0;

                for (auto i : Range(0, N))
                {
                    if (outlier[i]) continue;

                    auto& o  = obs[i];
                    auto& wp = wps[i];

                    if (o.depth > 0)
                    {
                        typename Saiga::Kernel::BAPoseStereo<T>::JacobiType Jrow;
                        Vec3 res;
                        Saiga::Kernel::BAPoseStereo<T>::evaluateResidualAndJacobian(camera, guess, wp, o.ip, o.depth,
                                                                                    res, Jrow, o.weight);

                        T c2     = res.squaredNorm();
                        chi2s[i] = c2;

#if 0
                        // Remove outliers
                        if (outerIt > 0 && innerIt == 0)
                        {
                            if (c2 > chi2Stereo)
                            {
                                outlier[i] = true;
                                continue;
                            }
                        }
#endif

                        if (robust)
                        {
                            //                            auto rw       = Kernel::huberWeight(deltaStereo, chi2s[i]);
                            //                            auto sqrtLoss = sqrt(rw(1));
                            //                            Jrow *= sqrtLoss;
                            //                            res *= sqrtLoss;
                            Saiga::Kernel::HuberRobustification<T> rob(deltaStereo);
                            rob.apply(res, Jrow);
                        }

                        chi2sum += res.squaredNorm();
                        //                        if constexpr (TriangularAdd)
                        JtJ += (Jrow.transpose() * Jrow).template triangularView<Eigen::Upper>();
                        //                        else
                        //                            JtJ += (Jrow.transpose() * Jrow);
                        Jtb += Jrow.transpose() * res;
                    }
                    else
                    {
                        typename Saiga::Kernel::BAPoseMono<T>::JacobiType Jrow;
                        Vec2 res;
                        Saiga::Kernel::BAPoseMono<T>::evaluateResidualAndJacobian(camera, guess, wp, o.ip, res, Jrow,
                                                                                  o.weight);
                        T c2     = res.squaredNorm();
                        chi2s[i] = c2;

#if 0
                        // Remove outliers
                        if (outerIt > 0 && innerIt == 0)
                        {
                            if (c2 > chi2Mono)
                            {
                                outlier[i] = true;
                                continue;
                            }
                        }
#endif

                        if (robust)
                        {
                            //                            auto rw       = Kernel::huberWeight(deltaMono, chi2s[i]);
                            //                            auto sqrtLoss = sqrt(rw(1));
                            //                            Jrow *= sqrtLoss;
                            //                            res *= sqrtLoss;
                            Saiga::Kernel::HuberRobustification<T> rob(deltaMono);
                            rob.apply(res, Jrow);
                        }

                        chi2sum += res.squaredNorm();
                        //                        if constexpr (TriangularAdd)
                        JtJ += (Jrow.transpose() * Jrow).template triangularView<Eigen::Upper>();
                        //                        else
                        //                            JtJ += (Jrow.transpose() * Jrow);
                        Jtb += Jrow.transpose() * res;
                    }
                }
                T deltaChi = lastChi2sum - chi2sum;

#if 0
                //                cout << outerIt << " Chi2: " << chi2sum << " Delta: " << deltaChi << endl;
                if (deltaChi < 0)
                {
                    // the error got worse :(
                    // -> discard step
                    guess = lastGuess;
                    break;
                }


                for (int k = 0; k < JtJ.rows(); ++k)
                {
                    auto& value = JtJ.diagonal()(k);
                    value       = value + 1e1 * value;
                    value       = std::clamp(value, 1e-6, 1e32);
                }

                //                cout << JtJ << endl;

                lastGuess                = guess;
#endif
                Eigen::Matrix<T, 6, 1> x = JtJ.template selfadjointView<Eigen::Upper>().ldlt().solve(Jtb);
                guess                    = SE3Type::exp(x) * guess;


                if (innerIt >= 1)
                {
                    // early termination if the error doesn't change
                    if (deltaChi < innerThreshold)
                    {
                        break;
                    }
                }

                lastChi2sum = chi2sum;
            }
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
                    auto res =
                        Saiga::Kernel::BAPoseStereo<T>::evaluateResidual(camera, guess, wp, o.ip, o.depth, o.weight);
                    chi2 = res.squaredNorm();
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
        }


        guess = pose;

        return inliers;
    }

    int optimizePoseRobust4(const AlignedVector<Vec4>& wps, const AlignedVector<Obs>& obs, AlignedVector<bool>& outlier,
                            SE3Type& guess, const CameraType& camera)
    {
        using MonoKernel   = Kernel::BAPoseMono<T, true>;
        using StereoKernel = Kernel::BAPoseStereo<T, true>;

        using MonoJ   = typename MonoKernel::JacobiType;
        using StereoJ = typename StereoKernel::JacobiType;

        StereoJ JrowS;
        MonoJ JrowM;
        JrowS.setZero();
        JrowM.setZero();

        int N       = wps.size();
        int inliers = 0;
        SE3Type pose;
        chi2s.resize(N);
        for (int it = 0; it < maxOuterIts; it++)
        {
            bool robust = it <= 2;
            pose        = guess;
            Eigen::Matrix<T, 8, 8> JtJ;
            Eigen::Matrix<T, 8, 1> Jtb;
            T lastChi2 = 0;

            for (int it = 0; it < maxInnerIts; ++it)
            {
                JtJ.setZero();
                Jtb.setZero();
                T chi2 = 0;

                for (size_t i = 0; i < wps.size(); ++i)
                {
                    if (outlier[i]) continue;

                    auto& o  = obs[i];
                    auto& wp = wps[i];

                    if (o.depth > 0)
                    {
                        Vec3 res;
                        StereoKernel::evaluateResidualAndJacobian(camera, guess, wp, o.ip, o.depth, res, JrowS,
                                                                  o.weight);
                        chi2s[i] = res.squaredNorm();

                        if (robust)
                        {
                            Saiga::Kernel::HuberRobustification<T> rob(deltaStereo);
                            rob.apply(res, JrowS);
                        }

                        chi2 += res.squaredNorm();
                        if constexpr (TriangularAdd)
                            JtJ += (JrowS.transpose() * JrowS).template triangularView<Eigen::Upper>();
                        else
                            JtJ += (JrowS.transpose() * JrowS);
                        Jtb += JrowS.transpose() * res;
                    }
                    else
                    {
                        Vec2 res;
                        MonoKernel::evaluateResidualAndJacobian(camera, guess, wp, o.ip, res, JrowM, o.weight);
                        chi2s[i] = res.squaredNorm();
                        if (robust)
                        {
                            Saiga::Kernel::HuberRobustification<T> rob(deltaMono);
                            rob.apply(res, JrowM);
                        }

                        chi2 += res.squaredNorm();
                        if constexpr (TriangularAdd)
                            JtJ += (JrowM.transpose() * JrowM).template triangularView<Eigen::Upper>();
                        else
                            JtJ += (JrowM.transpose() * JrowM);
                        Jtb += JrowM.transpose() * res;
                    }
                }
                Eigen::Matrix<T, 6, 6> test  = JtJ.template block<6, 6>(0, 0);
                Eigen::Matrix<T, 6, 1> testb = Jtb.template segment<6>(0);
                //                cout << JtJ << endl << endl;
                //                cout << Jtb << endl << endl;
                //                return 0;
                Eigen::Matrix<T, 6, 1> x = test.template selfadjointView<Eigen::Upper>().ldlt().solve(testb);
                //                Eigen::Matrix<T, 8, 1> x = JtJ.template block<6, 6>(0, 0).solve(Jtb);
                guess = SE3Type::exp(x) * guess;

                if (it >= 1)
                {
                    T deltaChi = lastChi2 - chi2;
                    // early termination if the error doesn't change
                    if (deltaChi < deltaChiEpsilon)
                    {
                        break;
                    }
                }
                lastChi2 = chi2;
            }

            T chi2sum = 0;

            inliers = 0;

            for (int i = 0; i < N; ++i)
            {
                auto& wp = wps[i];
                auto& o  = obs[i];

                T chi2;

                if (outlier[i])
                {
                    // we need to only recompute it for outliers
                    if (o.depth > 0)
                    {
                        auto res = StereoKernel::evaluateResidual(camera, guess, wp, o.ip, o.depth, o.weight);
                        chi2     = res.squaredNorm();
                    }
                    else
                    {
                        auto res = MonoKernel::evaluateResidual(camera, guess, wp, o.ip, o.weight);
                        chi2     = res.squaredNorm();
                    }
                }
                else
                {
                    chi2 = chi2s[i];
                }

                bool os = o.depth > 0 ? chi2 > chi2Stereo : chi2 > chi2Mono;

                if (!os)
                {
                    inliers++;
                    chi2sum += chi2;
                }
                outlier[i] = os;
            }
        }
        guess = pose;
        return inliers;
    }
};



}  // namespace Saiga
