/**
 * Copyright (c) 2017 Darius Rückert
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
SAIGA_VISION_API Vec3 linearRand(Vec3 low, Vec3 high);

SAIGA_VISION_API Vec3 ballRand(double radius);



template <typename MatrixType>
MatrixType gaussRandMatrix(double mean = 0, double stddev = 1)
{
    MatrixType M;
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j) M(i, j) = gaussRand(mean, stddev);
    return M;
}

template <typename MatrixType>
void setRandom(MatrixType& M)
{
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j) M(i, j) = sampleDouble(-1, 1);
}

SAIGA_VISION_API extern SE3 randomSE3();

}  // namespace Random
}  // namespace Saiga
