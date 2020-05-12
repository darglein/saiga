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
                auto pose_c_w_new = Sophus::se3_expd(p) * pose_c_w;
                //                auto pose_c_w_new = pose_c_w;
                //                Sophus::decoupled_inc(p, pose_c_w_new);

                return BundleAdjustment(intr, observation, pose_c_w_new, wp);
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



/// @brief Right Jacobian for decoupled SE(3)
///
/// For \f$ \exp(x) \in SE(3) \f$ provides a Jacobian that approximates the sum
/// under decoupled expmap with a right multiplication of decoupled expmap for
/// small \f$ \epsilon \f$.  Can be used to compute:  \f$ \exp(\phi + \epsilon)
/// \approx \exp(\phi) \exp(J_{\phi} \epsilon)\f$
/// @param[in] phi (6x1 vector)
/// @param[out] J_phi (6x6 matrix)
template <typename Derived1, typename Derived2>
inline void rightJacobianSE3Decoupled(const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 6);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 6, 6);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);

    J.setZero();

    Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
    rightJacobianSO3(omega, J.template bottomRightCorner<3, 3>());
    J.template topLeftCorner<3, 3>() = Sophus::SO3<Scalar>::exp(omega).inverse().matrix();
}



Vec3 RotatePointProject(const DSim3& pose, const Vec3& point, Matrix<double, 3, 7>* jacobian_pose = nullptr,
                        Matrix<double, 3, 3>* jacobian_point = nullptr)
{
    Vec3 p = pose * point;

    auto x = p(0);
    auto y = p(1);
    auto z = p(2);

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

        // Scale
        (*jacobian_pose)(0, 6) = x;
        (*jacobian_pose)(1, 6) = y;
        (*jacobian_pose)(2, 6) = z;
    }

    if (jacobian_point)
    {
        auto R            = pose.se3().so3().matrix();
        (*jacobian_point) = R;
        (*jacobian_point) *= pose.scale();
    }
    return p;
}


TEST(NumericDerivative, RotatePointProjectSim3)
{
    DSim3 pose_c_w = Random::randomDSim3();
    Vec3 wp        = Vec3::Random();

    Matrix<double, 3, 7> J_pose_1, J_pose_2;
    Matrix<double, 3, 3> J_point_1, J_point_2;
    Vec3 res1, res2;

    J_pose_1.setConstant(1000);

    res1 = RotatePointProject(pose_c_w, wp, &J_pose_1, &J_point_1);

    {
        Vec7 eps = Vec7::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto se3 = Sophus::dsim3_expd(p) * pose_c_w;
                return RotatePointProject(se3, wp);
            },
            eps, &J_pose_2);
    }
    {
        res2 = EvaluateNumeric([=](auto p) { return RotatePointProject(pose_c_w, p); }, wp, &J_point_2);
    }

    ExpectCloseRelative(res1, res2, 1e-10);
    ExpectCloseRelative(J_pose_1, J_pose_2, 1e-5);
    ExpectCloseRelative(J_point_1, J_point_2, 1e-10);
}


//
// Source:
// https://gitlab.com/VladyslavUsenko/basalt/-/blob/24e378a7a100d7d6f5133b34e16a09bb0cc98459/include/basalt/utils/nfr.h#L43-73
//
inline Sophus::Vector6d relPoseError(const Sophus::SE3d& T_i_j, const Sophus::SE3d& T_w_i, const Sophus::SE3d& T_w_j,
                                     Sophus::Matrix6d* d_res_d_T_w_i = nullptr,
                                     Sophus::Matrix6d* d_res_d_T_w_j = nullptr)
{
    Sophus::SE3d T_j_i   = T_w_j.inverse() * T_w_i;
    Sophus::Vector6d res = Sophus::se3_logd(T_i_j * T_j_i);

    if (d_res_d_T_w_i || d_res_d_T_w_j)
    {
        Sophus::Matrix6d J;
        Sophus::rightJacobianInvSE3Decoupled(res, J);

        Eigen::Matrix3d R = T_w_i.so3().inverse().matrix();

        Sophus::Matrix6d Adj;
        Adj.setZero();
        Adj.topLeftCorner<3, 3>()     = R;
        Adj.bottomRightCorner<3, 3>() = R;

        if (d_res_d_T_w_i)
        {
            *d_res_d_T_w_i = J * Adj;
        }

        if (d_res_d_T_w_j)
        {
            Adj.topRightCorner<3, 3>() = Sophus::SO3d::hat(T_j_i.inverse().translation()) * R;
            *d_res_d_T_w_j             = -J * Adj;
        }
    }

    return res;
}


TEST(NumericDerivative, RelativePose)
{
    SE3 pose_w_i = Random::randomSE3();
    SE3 pose_w_j = Random::randomSE3();
    SE3 pose_i_j = Sophus::se3_expd(Sophus::Vector6d::Random() / 100) * pose_w_i.inverse() * pose_w_j;

    Matrix<double, 6, 6> J_pose_i_1, J_pose_j_1, J_pose_i_2, J_pose_j_2;
    J_pose_i_1.setZero();
    J_pose_j_1.setZero();
    J_pose_i_2.setZero();
    J_pose_j_2.setZero();
    Vec6 res1, res2;

    res1 = relPoseError(pose_i_j, pose_w_i, pose_w_j, &J_pose_i_1, &J_pose_j_1);

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                //                auto pose_w_i_new = Sophus::se3_expd(p) * pose_w_i;

                auto pose_w_i_new = pose_w_i;
                Sophus::decoupled_inc(p, pose_w_i_new);

                return relPoseError(pose_i_j, pose_w_i_new, pose_w_j);
            },
            eps, &J_pose_i_2);
    }

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                //                auto pose_w_j_new = Sophus::se3_expd(p) * pose_w_j;
                auto pose_w_j_new = pose_w_j;
                Sophus::decoupled_inc(p, pose_w_j_new);

                return relPoseError(pose_i_j, pose_w_i, pose_w_j_new);
            },
            eps, &J_pose_j_2);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_pose_i_1, J_pose_i_2, 1e-5);
    ExpectCloseRelative(J_pose_j_1, J_pose_j_2, 1e-5);
}



inline Sophus::Vector7d relPoseError(const Sophus::DSim3<double>& T_i_j, const Sophus::DSim3<double>& T_w_i,
                                     const Sophus::DSim3<double>& T_w_j, Sophus::Matrix7d* d_res_d_T_w_i = nullptr,
                                     Sophus::Matrix7d* d_res_d_T_w_j = nullptr)
{
    Sophus::DSim3<double> T_j_i = T_w_j.inverse() * T_w_i;
    Sophus::Vector7d res        = Sophus::dsim3_logd(T_i_j * T_j_i);

    if (d_res_d_T_w_i || d_res_d_T_w_j)
    {
        Sophus::Matrix7d J;
        Sophus::rightJacobianInvDSim3Decoupled(res, J);

        Eigen::Matrix3d R = T_w_i.se3().so3().inverse().matrix();

        Sophus::Matrix7d Adj;
        Adj.setZero();
        Adj.topLeftCorner<3, 3>() = (1.0 / T_w_i.scale()) * R;
        Adj.block<3, 3>(3, 3)     = R;
        Adj(6, 6)                 = 1;

        if (d_res_d_T_w_i)
        {
            (*d_res_d_T_w_i) = J * Adj;
        }

        if (d_res_d_T_w_j)
        {
            Adj.block<3, 3>(0, 3) = Sophus::SO3d::hat(T_j_i.inverse().se3().translation()) * R;
            Adj.block<3, 1>(0, 6) = -T_j_i.inverse().se3().translation();
            (*d_res_d_T_w_j)      = -J * Adj;
        }
    }

    return res;
}


TEST(NumericDerivative, RelativePoseSim3)
{
    Sophus::DSim3d pose_w_j = Random::randomDSim3();
    Sophus::DSim3d pose_w_i = Random::randomDSim3();

    pose_w_i.scale()        = 0.7;
    pose_w_j.scale()        = 1.0;
    Sophus::DSim3d pose_i_j = Sophus::dsim3_expd(Sophus::Vector7d::Random() / 100) * pose_w_i.inverse() * pose_w_j;

    Matrix<double, 7, 7> J_pose_i_1, J_pose_j_1, J_pose_i_2, J_pose_j_2;
    J_pose_i_1.setZero();
    J_pose_j_1.setZero();
    J_pose_i_2.setZero();
    J_pose_j_2.setZero();
    Vec7 res1, res2;

    res1 = relPoseError(pose_i_j, pose_w_i, pose_w_j, &J_pose_i_1, &J_pose_j_1);

    {
        Vec7 eps = Vec7::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                //                auto pose_w_i_new = Sophus::dsim3_expd(p) * pose_w_i;

                auto pose_w_i_new = pose_w_i;
                decoupled_inc(p, pose_w_i_new);

                return relPoseError(pose_i_j, pose_w_i_new, pose_w_j);
            },
            eps, &J_pose_i_2, 1e-4);
    }

    {
        Vec7 eps = Vec7::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                //                auto pose_w_j_new = Sophus::dsim3_expd(p) * pose_w_j;
                auto pose_w_j_new = pose_w_j;
                decoupled_inc(p, pose_w_j_new);

                return relPoseError(pose_i_j, pose_w_i, pose_w_j_new);
            },
            eps, &J_pose_j_2, 1e-4);
    }

    ExpectCloseRelative(res1, res2, 1e-10);
    ExpectCloseRelative(J_pose_i_1, J_pose_i_2, 1e-8);
    ExpectCloseRelative(J_pose_j_1, J_pose_j_2, 1e-8);
}
}  // namespace Saiga
