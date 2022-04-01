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

    dual_quat.template head<4>() = transformation.unit_quaternion().coeffs();


    Vec3 t = transformation.translation();
    Quat t_as_quat(0, t(0), t(1), t(2));

    dual_quat.template tail<4>() = (t_as_quat * transformation.unit_quaternion()).coeffs();

    //    std::cout << dual_quat.template tail<4>().transpose() << std::endl;
    //    std::cout << (transformation.unit_quaternion() * t).transpose() << std::endl;

    //    exit(0);
    if (jacobian_transformation)
    {
        std::cout << 0.5 * transformation.unit_quaternion().matrix() << std::endl;
        std::cout << 0.5 * t_as_quat.matrix() << std::endl;
        jacobian_transformation->setZero();
    }

    return dual_quat;
}

template <typename T = double>
HD inline Vector<T, 3> RotateLie(const Quat q1, const Quat q2, Matrix<T, 3, 3>* jacobian_q1 = nullptr,
                                 Matrix<T, 3, 3>* jacobian_q2 = nullptr)
{
    // Quat result = Sophus::SO3d(q1 * q2).log();
    Vec3 result = Sophus::SO3d(q1 * q2).log();

    if (jacobian_q1)
    {
        std::cout << std::endl;
        std::cout << q1.coeffs().transpose() << std::endl;
        std::cout << q2.coeffs().transpose() << std::endl;
//        std::cout << result.coeffs().transpose() << std::endl;
        std::cout << std::endl;
        std::cout << q1.matrix() << std::endl;
        std::cout << q2.matrix() << std::endl;

        std::cout << result.matrix() << std::endl;
        std::cout << std::endl;
    }

//    return result.coeffs();
    return result;
}


TEST(NumericDerivative, RotateLie)
{
    Quat q1 = Random::randomQuat<double>();
    Quat q2 = Random::randomQuat<double>();

    Matrix<double, 3, 3> J_q1_1, J_q1_2;
    Matrix<double, 3, 3> J_q2_1, J_q2_2;
    Vec3 res1, res2, res3;

    res1 = RotateLie(q1, q2, &J_q1_1, &J_q2_1);

    {
        Vec3 eps = Vec3::Zero();
        res2     = EvaluateNumeric(
                [=](auto p)
                {
                Quat q = Sophus::SO3<double>::exp(p).unit_quaternion() * q1;
                return RotateLie(q, q2);
                },
                eps, &J_q1_2);
    }

    {
        Vec3 eps = Vec3::Zero();
        res3     = EvaluateNumeric(
                [=](auto p)
                {
                Quat q = Sophus::SO3<double>::exp(p).unit_quaternion() * q2;
                return RotateLie(q1, q);
                },
                eps, &J_q2_2);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(res1, res3, 1e-5);
    ExpectCloseRelative(J_q1_1, J_q1_2, 1e-5);
    ExpectCloseRelative(J_q2_1, J_q2_2, 1e-5);
    exit(0);
}



TEST(NumericDerivative, RotatePoint)
{
    SE3 transformation = Random::randomSE3();

    Matrix<double, 8, 6> J_trans_1, J_trans_2;
    Vec8 res1, res2;

    res1 = ToDual(transformation, &J_trans_1);

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
                [=](auto p)
                {
                SE3 se3 = Sophus::se3_expd(p) * transformation;
                return ToDual(se3);
                },
                eps, &J_trans_2);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_trans_1, J_trans_2, 1e-5);
}


}  // namespace Saiga
