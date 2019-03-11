/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/pgo/PGOConfig.h"
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

    static inline void evaluateResidual(const SE3Type& from, const SE3Type& to, const SE3Type& inverseMeasurement,
                                        ResidualType& res, T weight)
    {
        SE3Type res2 = from.inverse() * to * inverseMeasurement;
        res          = res2.log() * weight;
    }

    static inline void evaluateJacobian(const SE3Type& from, const SE3Type& to, const SE3Type& inverseMeasurement,
                                        PoseJacobiType& JrowFrom, PoseJacobiType& JrowTo, T weight)
    {
        JrowFrom = -from.inverse().Adj() * weight;
        JrowTo   = -JrowFrom;
    }

    static inline void evaluateResidualAndJacobian(const SE3Type& from, const SE3Type& to,
                                                   const SE3Type& inverseMeasurement, ResidualType& res,
                                                   PoseJacobiType& JrowFrom, PoseJacobiType& JrowTo, T weight)
    {
        evaluateResidual(from, to, inverseMeasurement, res, weight);
        evaluateJacobian(from, to, inverseMeasurement, JrowFrom, JrowTo, weight);
    }
};



}  // namespace Kernel
}  // namespace Saiga
