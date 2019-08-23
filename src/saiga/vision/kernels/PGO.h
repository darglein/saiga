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
// Works both for SE3 and Sim3
template <typename TransformationType = SE3>
struct PGO
{
    static constexpr int ResCount     = TransformationType::DoF;
    static constexpr int VarCountPose = TransformationType::DoF;

    using T                 = typename TransformationType::Scalar;
    using ResidualType      = Eigen::Matrix<T, ResCount, 1>;
    using ResidualBlockType = Eigen::Matrix<T, VarCountPose, 1>;
    using PoseJacobiType    = Eigen::Matrix<T, ResCount, VarCountPose, Eigen::RowMajor>;
    using PoseDiaBlockType  = Eigen::Matrix<T, VarCountPose, VarCountPose, Eigen::RowMajor>;


    static inline void evaluateResidual(const TransformationType& from, const TransformationType& to,
                                        const TransformationType& inverseMeasurement, ResidualType& res, T weight)
    {
#ifdef LSD_REL
        auto error_ = from.inverse() * to * inverseMeasurement;
#else
        auto error_ = inverseMeasurement.inverse() * from * to.inverse();
#endif
        res = error_.log() * weight;
    }

    static inline void evaluateJacobian(const TransformationType& from, const TransformationType& to,
                                        const TransformationType& inverseMeasurement, PoseJacobiType& JrowFrom,
                                        PoseJacobiType& JrowTo, T weight)
    {
#ifdef LSD_REL
        JrowTo   = from.inverse().Adj() * weight;
        JrowFrom = -JrowTo;
#else
        std::terminate();
#endif
    }

    static inline void evaluateResidualAndJacobian(const TransformationType& from, const TransformationType& to,
                                                   const TransformationType& inverseMeasurement, ResidualType& res,
                                                   PoseJacobiType& JrowFrom, PoseJacobiType& JrowTo, T weight)
    {
        evaluateResidual(from, to, inverseMeasurement, res, weight);
        evaluateJacobian(from, to, inverseMeasurement, JrowFrom, JrowTo, weight);
    }
};



}  // namespace Kernel
}  // namespace Saiga
