/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"

namespace Sophus
{
template <typename Derived>
inline SE3<typename Derived::Scalar> se3_expd(const Eigen::MatrixBase<Derived>& upsilon_omega)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 6);

    using Scalar = typename Derived::Scalar;

    return SE3<Scalar>(SO3<Scalar>::exp(upsilon_omega.template tail<3>()), upsilon_omega.template head<3>());
}

}  // namespace Sophus

namespace Saiga
{
Vec3 LinearFunction(const Vec2& x, Matrix<double, 3, 2>* jacobian = nullptr)
{
    Vec3 result;
    result(0) = 3.0 * x(0);
    result(1) = 7.0 * x(1);
    result(2) = 2.0 * x(0) + 5.0 * x(1);


    if (jacobian)
    {
        (*jacobian)(0, 0) = 3.0;
        (*jacobian)(0, 1) = 0;
        (*jacobian)(1, 0) = 0;
        (*jacobian)(1, 1) = 7.0;
        (*jacobian)(2, 0) = 2.0;
        (*jacobian)(2, 1) = 5.0;
    }
    return result;
}

TEST(NumericDerivative, Linear)
{
    Vec2 params = Vec2::Random();
    Matrix<double, 3, 2> J1, J2;

    Vec3 res1 = LinearFunction(params, &J1);
    Vec3 res2 = EvaluateNumeric([](auto p) { return LinearFunction(p); }, params, &J2);

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J1, J2, 1e-5);
}

Vec2 Polynomial(const Vec4& x, Matrix<double, 2, 4>* jacobian = nullptr)
{
    Vec2 result;
    result(0) = 1.0 * x(0) * x(0) * x(0) + 2.0 * x(0) * x(0) - 3.0 * x(0) + 4.0;

    result(1) = 0;
    result(1) += 2.0 * x(0) * x(0) * x(1) * x(2);
    result(1) += -3.0 * x(2) * x(1) * x(3) * x(3);



    if (jacobian)
    {
        (*jacobian)(0, 0) = 3.0 * x(0) * x(0) + 4.0 * x(0) - 3.0;
        (*jacobian)(0, 1) = 0;
        (*jacobian)(0, 2) = 0;
        (*jacobian)(0, 3) = 0;

        (*jacobian)(1, 0) = 4.0 * x(0) * x(1) * x(2);
        (*jacobian)(1, 1) = 2.0 * x(0) * x(0) * x(2) + -3.0 * x(2) * x(3) * x(3);
        (*jacobian)(1, 2) = 2.0 * x(0) * x(0) * x(1) + -3.0 * x(1) * x(3) * x(3);
        (*jacobian)(1, 3) = -6.0 * x(2) * x(1) * x(3);
    }
    return result;
}

TEST(NumericDerivative, Polynomial)
{
    Vec4 x = Vec4::Random();
    Matrix<double, 2, 4> J1, J2;

    auto res1 = Polynomial(x, &J1);
    auto res2 = EvaluateNumeric([](auto p) { return Polynomial(p); }, x, &J2);

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J1, J2, 1e-5);
}



Vec3 RotatePoint(const SE3& pose, const Vec3& point, Matrix<double, 3, 6>* jacobian_pose = nullptr,
                 Matrix<double, 3, 3>* jacobian_point = nullptr)
{
    Vec3 residual = pose * point;

    auto x = residual(0);
    auto y = residual(1);
    auto z = residual(2);

    if (jacobian_pose)
    {
        // translation
        (*jacobian_pose)(0, 0) = 1;
        (*jacobian_pose)(0, 1) = 0;
        (*jacobian_pose)(0, 2) = 0;
        (*jacobian_pose)(1, 0) = 0;
        (*jacobian_pose)(1, 1) = 1;
        (*jacobian_pose)(1, 2) = 0;
        (*jacobian_pose)(2, 0) = 0;
        (*jacobian_pose)(2, 1) = 0;
        (*jacobian_pose)(2, 2) = 1;

        // rotation
        (*jacobian_pose)(0, 3) = 0;
        (*jacobian_pose)(0, 4) = z;
        (*jacobian_pose)(0, 5) = -y;
        (*jacobian_pose)(1, 3) = -z;
        (*jacobian_pose)(1, 4) = 0;
        (*jacobian_pose)(1, 5) = x;
        (*jacobian_pose)(2, 3) = y;
        (*jacobian_pose)(2, 4) = -x;
        (*jacobian_pose)(2, 5) = 0;
    }

    if (jacobian_point)
    {
        auto R            = pose.so3().matrix();
        (*jacobian_point) = R;
    }
    return residual;
}


TEST(NumericDerivative, RotatePoint)
{
    SE3 pose_c_w = Random::randomSE3();
    Vec3 wp      = Vec3::Random();

    Matrix<double, 3, 6> J_pose_1, J_pose_2;
    Matrix<double, 3, 3> J_point_1, J_point_2;
    Vec3 res1, res2;

    res1 = RotatePoint(pose_c_w, wp, &J_pose_1, &J_point_1);

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto se3 = Sophus::se3_expd(p) * pose_c_w;
                return RotatePoint(se3, wp);
            },
            eps, &J_pose_2);
    }
    {
        res2 = EvaluateNumeric([=](auto p) { return RotatePoint(pose_c_w, p); }, wp, &J_point_2);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_pose_1, J_pose_2, 1e-5);
    ExpectCloseRelative(J_point_1, J_point_2, 1e-5);
}


Vec2 RotatePointProject(const SE3& pose, const Vec3& point, Matrix<double, 2, 6>* jacobian_pose = nullptr,
                        Matrix<double, 2, 3>* jacobian_point = nullptr)
{
    Vec3 p        = pose * point;
    Vec2 residual = Vec2(p(0) / p(2), p(1) / p(2));

    auto x = p(0);
    auto y = p(1);
    auto z = p(2);

    if (jacobian_pose)
    {
        // J of the rotation
        Matrix<double, 3, 3> J = Matrix<double, 3, 3>::Identity();

        // divion by z (this is a little verbose and can be simplified by a lot).
        (*jacobian_pose)(0, 0) = (J(0, 0) * z - x * J(2, 0)) / (z * z);
        (*jacobian_pose)(0, 1) = (J(0, 1) * z - x * J(2, 1)) / (z * z);
        (*jacobian_pose)(0, 2) = (J(0, 2) * z - x * J(2, 2)) / (z * z);
        (*jacobian_pose)(1, 0) = (J(1, 0) * z - y * J(2, 0)) / (z * z);
        (*jacobian_pose)(1, 1) = (J(1, 1) * z - y * J(2, 1)) / (z * z);
        (*jacobian_pose)(1, 2) = (J(1, 2) * z - y * J(2, 2)) / (z * z);

        J(0, 0) = 0;
        J(0, 1) = z;
        J(0, 2) = -y;
        J(1, 0) = -z;
        J(1, 1) = 0;
        J(1, 2) = x;
        J(2, 0) = y;
        J(2, 1) = -x;
        J(2, 2) = 0;

        // rotation
        (*jacobian_pose)(0, 3) = (J(0, 0) * z - x * J(2, 0)) / (z * z);
        (*jacobian_pose)(0, 4) = (J(0, 1) * z - x * J(2, 1)) / (z * z);
        (*jacobian_pose)(0, 5) = (J(0, 2) * z - x * J(2, 2)) / (z * z);
        (*jacobian_pose)(1, 3) = (J(1, 0) * z - y * J(2, 0)) / (z * z);
        (*jacobian_pose)(1, 4) = (J(1, 1) * z - y * J(2, 1)) / (z * z);
        (*jacobian_pose)(1, 5) = (J(1, 2) * z - y * J(2, 2)) / (z * z);
    }

    if (jacobian_point)
    {
        auto R = pose.so3().matrix();

        (*jacobian_point)(0, 0) = (R(0, 0) * z - x * R(2, 0)) / (z * z);
        (*jacobian_point)(0, 1) = (R(0, 1) * z - x * R(2, 1)) / (z * z);
        (*jacobian_point)(0, 2) = (R(0, 2) * z - x * R(2, 2)) / (z * z);
        (*jacobian_point)(1, 0) = (R(1, 0) * z - y * R(2, 0)) / (z * z);
        (*jacobian_point)(1, 1) = (R(1, 1) * z - y * R(2, 1)) / (z * z);
        (*jacobian_point)(1, 2) = (R(1, 2) * z - y * R(2, 2)) / (z * z);
    }
    return residual;
}


TEST(NumericDerivative, RotatePointProject)
{
    SE3 pose_c_w = Random::randomSE3();
    Vec3 wp      = Vec3::Random();

    Matrix<double, 2, 6> J_pose_1, J_pose_2;
    Matrix<double, 2, 3> J_point_1, J_point_2;
    Vec2 res1, res2;

    res1 = RotatePointProject(pose_c_w, wp, &J_pose_1, &J_point_1);

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto se3 = Sophus::se3_expd(p) * pose_c_w;
                //                auto se3 = SE3::exp(p) * pose_c_w;
                return RotatePointProject(se3, wp);
            },
            eps, &J_pose_2);
    }
    {
        res2 = EvaluateNumeric([=](auto p) { return RotatePointProject(pose_c_w, p); }, wp, &J_point_2);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_pose_1, J_pose_2, 1e-5);
    ExpectCloseRelative(J_point_1, J_point_2, 1e-5);
}


Vec2 BundleAdjustment(const Intrinsics4& camera, const Vec2& observation, const SE3& pose, const Vec3& point,
                      Matrix<double, 2, 6>* jacobian_pose = nullptr, Matrix<double, 2, 3>* jacobian_point = nullptr)
{
    Vec3 p      = pose * point;
    Vec2 p_by_z = Vec2(p(0) / p(2), p(1) / p(2));

    Vec2 residual;
    residual(0) = camera.fx * p_by_z(0) + camera.cx - observation(0);
    residual(1) = camera.fy * p_by_z(1) + camera.cy - observation(1);

    auto x = p(0);
    auto y = p(1);
    auto z = p(2);

    if (jacobian_pose)
    {
        // J of the rotation
        Matrix<double, 3, 3> J = Matrix<double, 3, 3>::Identity();

        // divion by z (this is a little verbose and can be simplified by a lot).
        (*jacobian_pose)(0, 0) = (J(0, 0) * z - x * J(2, 0)) / (z * z);
        (*jacobian_pose)(0, 1) = (J(0, 1) * z - x * J(2, 1)) / (z * z);
        (*jacobian_pose)(0, 2) = (J(0, 2) * z - x * J(2, 2)) / (z * z);
        (*jacobian_pose)(1, 0) = (J(1, 0) * z - y * J(2, 0)) / (z * z);
        (*jacobian_pose)(1, 1) = (J(1, 1) * z - y * J(2, 1)) / (z * z);
        (*jacobian_pose)(1, 2) = (J(1, 2) * z - y * J(2, 2)) / (z * z);

        (*jacobian_pose)(0, 0) *= camera.fx;
        (*jacobian_pose)(0, 1) *= camera.fx;
        (*jacobian_pose)(0, 2) *= camera.fx;
        (*jacobian_pose)(1, 0) *= camera.fy;
        (*jacobian_pose)(1, 1) *= camera.fy;
        (*jacobian_pose)(1, 2) *= camera.fy;

        J(0, 0) = 0;
        J(0, 1) = z;
        J(0, 2) = -y;
        J(1, 0) = -z;
        J(1, 1) = 0;
        J(1, 2) = x;
        J(2, 0) = y;
        J(2, 1) = -x;
        J(2, 2) = 0;

        // rotation
        (*jacobian_pose)(0, 3) = (J(0, 0) * z - x * J(2, 0)) / (z * z);
        (*jacobian_pose)(0, 4) = (J(0, 1) * z - x * J(2, 1)) / (z * z);
        (*jacobian_pose)(0, 5) = (J(0, 2) * z - x * J(2, 2)) / (z * z);
        (*jacobian_pose)(1, 3) = (J(1, 0) * z - y * J(2, 0)) / (z * z);
        (*jacobian_pose)(1, 4) = (J(1, 1) * z - y * J(2, 1)) / (z * z);
        (*jacobian_pose)(1, 5) = (J(1, 2) * z - y * J(2, 2)) / (z * z);

        (*jacobian_pose)(0, 3) *= camera.fx;
        (*jacobian_pose)(0, 4) *= camera.fx;
        (*jacobian_pose)(0, 5) *= camera.fx;
        (*jacobian_pose)(1, 3) *= camera.fy;
        (*jacobian_pose)(1, 4) *= camera.fy;
        (*jacobian_pose)(1, 5) *= camera.fy;
    }

    if (jacobian_point)
    {
        auto R = pose.so3().matrix();

        (*jacobian_point)(0, 0) = (R(0, 0) * z - x * R(2, 0)) / (z * z);
        (*jacobian_point)(0, 1) = (R(0, 1) * z - x * R(2, 1)) / (z * z);
        (*jacobian_point)(0, 2) = (R(0, 2) * z - x * R(2, 2)) / (z * z);
        (*jacobian_point)(1, 0) = (R(1, 0) * z - y * R(2, 0)) / (z * z);
        (*jacobian_point)(1, 1) = (R(1, 1) * z - y * R(2, 1)) / (z * z);
        (*jacobian_point)(1, 2) = (R(1, 2) * z - y * R(2, 2)) / (z * z);

        (*jacobian_point)(0, 0) *= camera.fx;
        (*jacobian_point)(0, 1) *= camera.fx;
        (*jacobian_point)(0, 2) *= camera.fx;
        (*jacobian_point)(1, 0) *= camera.fy;
        (*jacobian_point)(1, 1) *= camera.fy;
        (*jacobian_point)(1, 2) *= camera.fy;
    }
    return residual;
}


Vec2 BundleAdjustmentCompact(const Intrinsics4& camera, const Vec2& observation, const SE3& pose, const Vec3& point,
                             Matrix<double, 2, 6>* jacobian_pose  = nullptr,
                             Matrix<double, 2, 3>* jacobian_point = nullptr)
{
    Vec3 p      = pose * point;
    Vec2 p_by_z = Vec2(p(0) / p(2), p(1) / p(2));

    Vec2 residual;
    residual(0) = camera.fx * p_by_z(0) + camera.cx - observation(0);
    residual(1) = camera.fy * p_by_z(1) + camera.cy - observation(1);

    auto x     = p(0);
    auto y     = p(1);
    auto z     = p(2);
    auto zz    = z * z;
    auto zinv  = 1 / z;
    auto zzinv = 1 / zz;
    if (jacobian_pose)
    {
        // divion by z (this is a little verbose and can be simplified by a lot).
        (*jacobian_pose)(0, 0) = zinv;
        (*jacobian_pose)(0, 1) = 0;
        (*jacobian_pose)(0, 2) = -x * zzinv;
        (*jacobian_pose)(1, 0) = 0;
        (*jacobian_pose)(1, 1) = zinv;
        (*jacobian_pose)(1, 2) = -y * zzinv;


        // rotation
        (*jacobian_pose)(0, 3) = -y * x * zzinv;
        (*jacobian_pose)(0, 4) = (1 + (x * x) * zzinv);
        (*jacobian_pose)(0, 5) = -y * zinv;
        (*jacobian_pose)(1, 3) = (-1 - (y * y) * zzinv);
        (*jacobian_pose)(1, 4) = x * y * zzinv;
        (*jacobian_pose)(1, 5) = x * zinv;

        (*jacobian_pose).row(0) *= camera.fx;
        (*jacobian_pose).row(1) *= camera.fy;
    }

    if (jacobian_point)
    {
        auto R = pose.so3().matrix();

        (*jacobian_point)(0, 0) = (R(0, 0) - p_by_z(0) * R(2, 0)) * zinv;
        (*jacobian_point)(0, 1) = (R(0, 1) - p_by_z(0) * R(2, 1)) * zinv;
        (*jacobian_point)(0, 2) = (R(0, 2) - p_by_z(0) * R(2, 2)) * zinv;
        (*jacobian_point)(1, 0) = (R(1, 0) - p_by_z(1) * R(2, 0)) * zinv;
        (*jacobian_point)(1, 1) = (R(1, 1) - p_by_z(1) * R(2, 1)) * zinv;
        (*jacobian_point)(1, 2) = (R(1, 2) - p_by_z(1) * R(2, 2)) * zinv;

        (*jacobian_point).row(0) *= camera.fx;
        (*jacobian_point).row(1) *= camera.fy;
    }
    return residual;
}



TEST(NumericDerivative, BundleAdjustment)
{
    SE3 pose_c_w = Random::randomSE3();
    Vec3 wp      = Vec3::Random();
    Intrinsics4 intr;
    intr.coeffs(Vec4::Random());

    Vec2 projection  = intr.project(pose_c_w * wp);
    Vec2 observation = projection + Vec2::Random() * 0.1;

    Matrix<double, 2, 6> J_pose_1, J_pose_2, J_pose_3;
    Matrix<double, 2, 3> J_point_1, J_point_2, J_point_3;
    Vec2 res1, res2, res3;

    res1 = BundleAdjustment(intr, observation, pose_c_w, wp, &J_pose_1, &J_point_1);
    res3 = BundleAdjustmentCompact(intr, observation, pose_c_w, wp, &J_pose_3, &J_point_3);

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto se3 = Sophus::se3_expd(p) * pose_c_w;
                return BundleAdjustment(intr, observation, se3, wp);
            },
            eps, &J_pose_2);
    }
    {
        res2 =
            EvaluateNumeric([=](auto p) { return BundleAdjustment(intr, observation, pose_c_w, p); }, wp, &J_point_2);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(res2, res3, 1e-5);
    ExpectCloseRelative(J_pose_1, J_pose_2, 1e-5);
    ExpectCloseRelative(J_pose_2, J_pose_3, 1e-5);
    ExpectCloseRelative(J_point_1, J_point_2, 1e-5);
    ExpectCloseRelative(J_point_2, J_point_3, 1e-5);
}



}  // namespace Saiga
