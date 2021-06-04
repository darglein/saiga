/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VisionTypes.h"

#include "VisionIncludes.h"

namespace Saiga
{




Mat3 enforceRank2(const Mat3& M)
{
    // enforce it with svd
    // det(F)=0
    auto svde = M.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto svs  = svde.singularValues();
    // set third singular value to 0
    svs(2) = 0;

    Eigen::DiagonalMatrix<double, 3> sigma;
    sigma.diagonal() = svs;

    Mat3 u = svde.matrixU();
    Mat3 v = svde.matrixV();
    auto F = u * sigma * v.transpose();
    return F;
}

Vec3 infinityVec3()
{
    return Vec3(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity());
}

}  // namespace Saiga
