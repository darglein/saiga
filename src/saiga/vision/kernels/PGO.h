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
//        SE3Type res2 = from.inverse() * to * inverseMeasurement;
//        res          = res2.log() * weight;
#ifdef LSD_REL
        auto error_ = from.inverse() * to * inverseMeasurement;
#else
        auto error_ = inverseMeasurement.inverse() * from * to.inverse();
#endif
        res = error_.log() * weight;
    }

    static inline void evaluateJacobian(const SE3Type& from, const SE3Type& to, const SE3Type& inverseMeasurement,
                                        PoseJacobiType& JrowFrom, PoseJacobiType& JrowTo, T weight)
    {
#ifdef LSD_REL
        JrowFrom = -from.inverse().Adj() * weight;
        JrowTo   = -JrowFrom;
#else
        auto delta  = 1e-9;
        auto scalar = 1 / (2 * delta);

        for (auto i : Range(0, 6))
        {
            // add small step along the unit vector in each dimension
            Vec6 unitDelta = Vec6::Zero();
            unitDelta[i]   = delta;

            SE3 n1cpy = from;
            SE3 n2cpy = to;

            n1cpy = SE3::exp(unitDelta) * n1cpy;

            ResidualType errorPlus;
            evaluateResidual(n1cpy, n2cpy, inverseMeasurement, errorPlus, weight);

            n1cpy        = from;
            unitDelta[i] = -delta;
            n1cpy        = SE3::exp(unitDelta) * n1cpy;


            ResidualType errorMinus;
            evaluateResidual(n1cpy, n2cpy, inverseMeasurement, errorMinus, weight);

            JrowFrom.col(i) = scalar * (errorPlus - errorMinus);
        }

        for (auto i : Range(0, 6))
        {
            // add small step along the unit vector in each dimension
            Vec6 unitDelta = Vec6::Zero();
            unitDelta[i]   = delta;

            SE3 n1cpy = from;
            SE3 n2cpy = to;

            n2cpy = SE3::exp(unitDelta) * n2cpy;

            ResidualType errorPlus;
            evaluateResidual(n1cpy, n2cpy, inverseMeasurement, errorPlus, weight);

            n2cpy        = to;
            unitDelta[i] = -delta;
            n2cpy        = SE3::exp(unitDelta) * n2cpy;


            ResidualType errorMinus;
            evaluateResidual(n1cpy, n2cpy, inverseMeasurement, errorMinus, weight);

            JrowTo.col(i) = scalar * (errorPlus - errorMinus);
        }
//        std::cout << std::endl;
//        std::cout << JrowFrom << std::endl << std::endl;
//        std::cout << JrowTo << std::endl << std::endl;
//        exit(0);
#endif
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
