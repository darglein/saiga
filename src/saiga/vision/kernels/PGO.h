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
                                        ResidualType& res)
    {
#ifdef LSD_REL
        SE3Type res2 = from.inverse() * to * inverseMeasurement;
#else
        SE3Type res2 = to * from.inverse() * inverseMeasurement;
#endif
        res = res2.log();
    }

    static inline void evaluateJacobian(const SE3Type& from, const SE3Type& to, const SE3Type& inverseMeasurement,
                                        PoseJacobiType& JrowFrom, PoseJacobiType& JrowTo)
    {
#ifdef LSD_REL
        JrowFrom = from.inverse().Adj();
        JrowTo   = -JrowFrom;
#else
        JrowFrom     = to.Adj();
        JrowTo       = -JrowFrom;
#endif
    }

    static inline void evaluateResidualAndJacobian(const SE3Type& from, const SE3Type& to,
                                                   const SE3Type& inverseMeasurement, ResidualType& res,
                                                   PoseJacobiType& JrowFrom, PoseJacobiType& JrowTo)
    {
        evaluateResidual(from, to, inverseMeasurement, res);
        evaluateJacobian(from, to, inverseMeasurement, JrowFrom, JrowTo);

        //        JrowFrom = to.Adj();
        //        JrowTo   = -JrowFrom;

        //        JrowFrom = from.inverse().Adj();
        //        JrowTo   = -JrowFrom;
    }
};



}  // namespace Kernel
}  // namespace Saiga
