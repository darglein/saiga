/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/Range.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/vision/VisionTypes.h"
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

template <typename T>
struct PoseOptimizationScene
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StereoCamera4Base<T> K;
    double scale = 1;
    Sophus::SE3<T> pose;
    AlignedVector<Vec3> wps;
    AlignedVector<ObsBase<T>> obs;
    AlignedVector<int> outlier;

    Sophus::SE3<T> prediction;
    double weight_rotation    = 0;
    double weight_translation = 0;


    double chi2()
    {
        double result = 0;
        for (int i = 0; i < obs.size(); ++i)
        {
            Vec2 ip      = K.project(pose * wps[i]);
            double error = (ip - obs[i].ip).squaredNorm();
            result += error;
        }
        result += predictionError();
        return result;
    }
    double rms()
    {
        double result = 0;
        for (int i = 0; i < obs.size(); ++i)
        {
            Vec2 ip      = K.project(pose * wps[i]);
            double error = (ip - obs[i].ip).squaredNorm();
            result += error;
        }
        result /= obs.size();
        result = sqrt(result);
        return result;
    }

    double predictionError()
    {
        Sophus::SE3d T_j_i   = prediction.inverse() * pose;
        Sophus::Vector6d res = Sophus::se3_logd(T_j_i);

        res.template segment<3>(0) *= T(weight_translation);
        res.template segment<3>(3) *= T(weight_rotation);

        Vec6 residual = res;
        return residual.squaredNorm();
    }
};


}  // namespace Saiga
