/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
namespace Kernel
{
template <typename T>
struct PGO
{
    static constexpr int ResCount     = 6;
    static constexpr int VarCountPose = 6;

    using ResidualType      = Eigen::Matrix<T, ResCount, 1>;
    using ResidualBlockType = Eigen::Matrix<T, VarCountPose, 1>;
    using PoseJacobiType    = Eigen::Matrix<T, ResCount, VarCountPose, Eigen::RowMajor>;
    using PoseDiaBlockType  = Eigen::Matrix<T, VarCountPose, VarCountPose, Eigen::RowMajor>;


    using SE3Type = Sophus::SE3<T>;

    static inline ResidualType evaluateResidual(const SE3Type& from, const SE3Type& to,
                                                const SE3Type& inverseMeasurement)
    {
        SE3Type res = from.inverse() * to * inverseMeasurement;
        return res.log();
    }

    static inline void evaluateResidualAndJacobian(const SE3Type& from, const SE3Type& to,
                                                   const SE3Type& inverseMeasurement, ResidualType& res,
                                                   PoseJacobiType& JrowFrom, PoseJacobiType& JrowTo)
    {
        res      = evaluateResidual(from, to, inverseMeasurement);
        JrowFrom = from.inverse().Adj();
        JrowTo   = -JrowFrom;
    }
};



}  // namespace Kernel
}  // namespace Saiga
