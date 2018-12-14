/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

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
};

using Obs = ObsBase<double>;

template <typename T>
struct SAIGA_GLOBAL RobustPoseOptimization
{
    using CameraType = StereoCamera4Base<T>;
    using SE3Type    = Sophus::SE3<T>;
    using Vec3       = Eigen::Matrix<T, 3, 1>;
    using Vec2       = Eigen::Matrix<T, 2, 1>;
    using Obs        = ObsBase<T>;

    std::vector<T> chi2Mono   = {5.991, 5.991, 5.991, 5.991};
    std::vector<T> chi2Stereo = {7.815, 7.815, 7.815, 7.815};
    std::vector<int> its      = {10, 10, 10, 10};
    T deltaMono               = sqrt(5.991);
    T deltaStereo             = sqrt(7.815);
    T deltaChiEpsilon         = 1e-4;
    size_t iterations         = 4;

    int optimizePoseRobust(const AlignedVector<Vec3>& wps, const AlignedVector<Obs>& obs, AlignedVector<bool>& outlier,
                           SE3Type& guess, const CameraType& camera);

   private:
    std::vector<T> chi2s;
};

template <typename T>
int RobustPoseOptimization<T>::optimizePoseRobust(const AlignedVector<Vec3>& wps, const AlignedVector<Obs>& obs,
                                                  AlignedVector<bool>& outlier, SE3Type& guess,
                                                  const CameraType& camera)
{
    size_t N    = wps.size();
    int inliers = 0;
    SE3Type pose;
    chi2s.resize(N);
    for (size_t it = 0; it < iterations; it++)
    {
        bool robust   = it <= 2;
        pose          = guess;
        auto innerIts = its[it];
        Eigen::Matrix<T, 6, 6> JtJ;
        Eigen::Matrix<T, 6, 1> Jtb;
        T lastChi2 = 0;

        for (int it = 0; it < innerIts; ++it)
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
                    Eigen::Matrix<T, 3, 6> Jrow;
                    Vec3 res;
                    Saiga::Kernel::BAPoseStereo<T>::evaluateResidualAndJacobian(camera, guess, wp, o.ip, o.depth, res,
                                                                                Jrow, o.weight);
                    chi2s[i] = res.squaredNorm();

                    if (robust)
                    {
                        Saiga::Kernel::HuberRobustification<T> rob(deltaStereo);
                        rob.apply(res, Jrow);
                    }

                    chi2 += res.squaredNorm();
                    JtJ += (Jrow.transpose() * Jrow).template triangularView<Eigen::Upper>();
                    Jtb += Jrow.transpose() * res;
                }
                else
                {
                    typename Saiga::Kernel::BAPoseMono<T>::PoseJacobiType Jrow;
                    Vec2 res;
                    Saiga::Kernel::BAPoseMono<T>::evaluateResidualAndJacobian(camera, guess, wp, o.ip, res, Jrow,
                                                                              o.weight);
                    chi2s[i] = res.squaredNorm();
                    if (robust)
                    {
                        Saiga::Kernel::HuberRobustification<T> rob(deltaMono);
                        rob.apply(res, Jrow);
                    }

                    chi2 += res.squaredNorm();
                    JtJ += (Jrow.transpose() * Jrow).template triangularView<Eigen::Upper>();
                    Jtb += Jrow.transpose() * res;
                }
            }
            Eigen::Matrix<T, 6, 1> x = JtJ.template selfadjointView<Eigen::Upper>().ldlt().solve(Jtb);
            guess                    = SE3Type::exp(x) * guess;

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

        for (size_t i = 0; i < N; ++i)
        {
            auto& wp = wps[i];
            auto& o  = obs[i];

            T chi2;

            if (outlier[i])
            {
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
            }
            else
            {
                chi2 = chi2s[i];
            }

            bool os = o.depth > 0 ? chi2 > chi2Stereo[it] : chi2 > chi2Mono[it];

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


}  // namespace Saiga
