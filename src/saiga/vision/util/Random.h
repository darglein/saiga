/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/random.h"
#include "saiga/vision/VisionIncludes.h"

namespace Saiga
{
namespace Random
{
// template <typename MatrixType>
// MatrixType MatrixGauss(double mean = 0, double stddev = 1)
//{
//    MatrixType M;
//    for (int i = 0; i < M.rows(); ++i)
//        for (int j = 0; j < M.cols(); ++j) M(i, j) = gaussRand(mean, stddev);
//    return M;
//}

// template <typename MatrixType>
// void setRandom(MatrixType& M)
//{
//    for (int i = 0; i < M.rows(); ++i)
//        for (int j = 0; j < M.cols(); ++j) M(i, j) = sampleDouble(-1, 1);
//}

SAIGA_VISION_API extern SE3 randomSE3();
SAIGA_VISION_API extern Sophus::Sim3d randomSim3();
SAIGA_VISION_API extern DSim3 randomDSim3();

/**
 * Add random noise to a SE3. Usefull to test iterative solvers if they converge back to the original value.
 */
template <typename T>
SE3 JitterPose(const SE3& pose, T translation_sdev = T(0.01), T rotation_sdev = T(0.002))
{
    SE3 result = pose;
    result.translation() += Random::MatrixGauss<Vec3>(0, translation_sdev);
    Quat q = result.unit_quaternion();
    q.coeffs() += Random::MatrixGauss<Vec4>(0, rotation_sdev);
    q.normalize();
    result.setQuaternion(q);
    return result;
}


}  // namespace Random
}  // namespace Saiga
