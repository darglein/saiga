/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vision/imu/Solver.h"
#include "saiga/vision/kernels/BA.h"
#include "saiga/vision/kernels/PGO.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"
namespace Saiga
{


template <typename T = double>
HD inline Vector<T, 8> ToDual(const Sophus::SE3<T>& transformation, Matrix<T, 8, 6>* jacobian_transformation = nullptr)
{
    Vector<T, 8> dual_quat;

//    dual_quat.template head<4>() = transformation.unit_quaternion().coeffs();
//
//
//    Vec3 t = transformation.translation();
//    Quat t_as_quat(0, t(0), t(1), t(2));
//    dual_quat.template tail<4>() = (1.f / 2.f) * (t_as_quat * transformation.unit_quaternion()).coeffs();
//
//    if (jacobian_transformation)
//    {
//        jacobian_transformation->setZero();
//    }

    return dual_quat;
}

TEST(NumericDerivative, RotatePoint)
{
    SE3 transformation = Random::randomSE3();

//    Matrix<double, 8, 6> J_trans_1, J_trans_2;
//    Vec8 res1, res2;
//
//    res1 = ToDual(transformation, &J_trans_1);
//
//    {
//        Vec6 eps = Vec6::Zero();
//        res2     = EvaluateNumeric(
//                [=](auto p)
//                {
//                SE3 se3 = Sophus::se3_expd(p) * transformation;
//                return ToDual(se3);
//                },
//                eps, &J_trans_2);
//    }
//
//    ExpectCloseRelative(res1, res2, 1e-5);
//    ExpectCloseRelative(J_trans_1, J_trans_2, 1e-5);
}


}  // namespace Saiga
