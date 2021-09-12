/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


@file
@brief Useful utilities to work with SO(3) and SE(3) groups from Sophus.
*/

#pragma once

#include "saiga/core/sophus/SophusSelector.h"

namespace Sophus
{
/// @brief Decoupled version of logmap for SE(3)
///
/// For SE(3) element vector
/// \f[
/// \begin{pmatrix} R & t \\ 0 & 1 \end{pmatrix} \in SE(3),
/// \f]
/// returns \f$ (t, \log(R)) \in \mathbb{R}^6 \f$. Here rotation is not coupled
/// with translation.
///
/// @param[in] SE(3) member
/// @return tangent vector (6x1 vector)
template <typename Scalar>
inline typename SE3<Scalar>::Tangent se3_logd(const SE3<Scalar>& se3)
{
    typename SE3<Scalar>::Tangent upsilon_omega;
    upsilon_omega.template head<3>() = se3.translation();
    upsilon_omega.template tail<3>() = se3.so3().log();

    return upsilon_omega;
}

template <typename Scalar>
inline typename Sim3<Scalar>::Tangent sim3_logd(const Sim3<Scalar>& se3)
{
    typename Sim3<Scalar>::Tangent upsilon_omega;
    upsilon_omega.template tail<4>() = se3.rxso3().log();
    upsilon_omega.template head<3>() = se3.translation();

    return upsilon_omega;
}

template <typename Scalar>
inline Eigen::Matrix<Scalar, 7, 1> dsim3_logd(const DSim3<Scalar>& sim3)
{
    Eigen::Matrix<Scalar, 7, 1> upsilon_omega_scale;
    upsilon_omega_scale.template head<6>() = se3_logd(sim3.se3());
    upsilon_omega_scale(6)                 = log(sim3.scale());
    return upsilon_omega_scale;
}


/// @brief Decoupled version of expmap for SE(3)
///
/// For tangent vector \f$ (\upsilon, \omega) \in \mathbb{R}^6 \f$ returns
/// \f[
/// \begin{pmatrix} \exp(\omega) & \upsilon \\ 0 & 1 \end{pmatrix} \in SE(3),
/// \f]
/// where \f$ \exp(\omega) \in SO(3) \f$. Here rotation is not coupled with
/// translation.
///
/// @param[in] tangent vector (6x1 vector)
/// @return  SE(3) member
template <typename Derived>
HD inline SE3<typename Derived::Scalar> se3_expd(const Eigen::MatrixBase<Derived>& upsilon_omega)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 6);

    using Scalar = typename Derived::Scalar;

    return SE3<Scalar>(SO3<Scalar>::exp(upsilon_omega.template tail<3>()), upsilon_omega.template head<3>());
}

template <typename Derived>
HD inline DSim3<typename Derived::Scalar> dsim3_expd(const Eigen::MatrixBase<Derived>& upsilon_omega)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 7);

    using Scalar = typename Derived::Scalar;
    return DSim3<Scalar>(se3_expd(upsilon_omega.template head<6>()), exp(upsilon_omega(6)));
}

/// @brief Right Jacobian for SO(3)
///
/// For \f$ \exp(x) \in SO(3) \f$ provides a Jacobian that approximates the sum
/// under expmap with a right multiplication of expmap for small \f$ \epsilon
/// \f$.  Can be used to compute:  \f$ \exp(\phi + \epsilon) \approx \exp(\phi)
/// \exp(J_{\phi} \epsilon)\f$
/// @param[in] phi (3x1 vector)
/// @param[out] J_phi (3x3 matrix)
template <typename Derived1, typename Derived2>
inline void rightJacobianSO3(const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);

    Scalar phi_norm2 = phi.squaredNorm();
    Scalar phi_norm  = std::sqrt(phi_norm2);
    Scalar phi_norm3 = phi_norm2 * phi_norm;

    J.setIdentity();

    if (Sophus::Constants<Scalar>::epsilon() < phi_norm)
    {
        Eigen::Matrix<Scalar, 3, 3> phi_hat  = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J -= phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
        J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
    }
}


/// @brief Left Jacobian for SO(3)
///
/// For \f$ \exp(x) \in SO(3) \f$ provides a Jacobian that approximates the sum
/// under expmap with a left multiplication of expmap for small \f$ \epsilon
/// \f$.  Can be used to compute:  \f$ \exp(\phi + \epsilon) \approx
/// \exp(J_{\phi} \epsilon) \exp(\phi) \f$
/// @param[in] phi (3x1 vector)
/// @param[out] J_phi (3x3 matrix)
template <typename Derived1, typename Derived2>
inline void leftJacobianSO3(const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);

    Scalar phi_norm2 = phi.squaredNorm();
    Scalar phi_norm  = std::sqrt(phi_norm2);
    Scalar phi_norm3 = phi_norm2 * phi_norm;

    J.setIdentity();

    if (Sophus::Constants<Scalar>::epsilon() < phi_norm)
    {
        Eigen::Matrix<Scalar, 3, 3> phi_hat  = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J += phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
        J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
    }
}

/// @brief Right Inverse Jacobian for SO(3)
///
/// For \f$ \exp(x) \in SO(3) \f$ provides an inverse Jacobian that approximates
/// the logmap of the right multiplication of expmap of the arguments with a sum
/// for small \f$ \epsilon \f$.  Can be used to compute:  \f$ \log
/// (\exp(\phi) \exp(\epsilon)) \approx \phi + J_{\phi} \epsilon\f$
/// @param[in] phi (3x1 vector)
/// @param[out] J_phi (3x3 matrix)
template <typename Derived1, typename Derived2>
inline void rightJacobianInvSO3(const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);

    Scalar phi_norm2 = phi.squaredNorm();
    Scalar phi_norm  = std::sqrt(phi_norm2);

    J.setIdentity();

    if (Sophus::Constants<Scalar>::epsilon() < phi_norm)
    {
        Eigen::Matrix<Scalar, 3, 3> phi_hat  = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J += phi_hat / 2;
        J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) / (2 * phi_norm * std::sin(phi_norm)));
    }
}


/// @brief Left Inverse Jacobian for SO(3)
///
/// For \f$ \exp(x) \in SO(3) \f$ provides an inverse Jacobian that approximates
/// the logmap of the left multiplication of expmap of the arguments with a sum
/// for small \f$ \epsilon \f$.  Can be used to compute:  \f$ \log
/// (\exp(\epsilon) \exp(\phi)) \approx \phi + J_{\phi} \epsilon\f$
/// @param[in] phi (3x1 vector)
/// @param[out] J_phi (3x3 matrix)
template <typename Derived1, typename Derived2>
inline void leftJacobianInvSO3(const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);

    Scalar phi_norm2 = phi.squaredNorm();
    Scalar phi_norm  = std::sqrt(phi_norm2);

    J.setIdentity();

    if (Sophus::Constants<Scalar>::epsilon() < phi_norm)
    {
        Eigen::Matrix<Scalar, 3, 3> phi_hat  = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J -= phi_hat / 2;
        J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) / (2 * phi_norm * std::sin(phi_norm)));
    }
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

/// @brief Right Inverse Jacobian for decoupled SE(3)
///
/// For \f$ \exp(x) \in SE(3) \f$ provides an inverse Jacobian that approximates
/// the decoupled logmap of the right multiplication of the decoupled expmap of
/// the arguments with a sum for small \f$ \epsilon \f$.  Can be used to
/// compute:  \f$ \log
/// (\exp(\phi) \exp(\epsilon)) \approx \phi + J_{\phi} \epsilon\f$
/// @param[in] phi (6x1 vector)
/// @param[out] J_phi (6x6 matrix)
template <typename Derived1, typename Derived2>
inline void rightJacobianInvSE3Decoupled(const Eigen::MatrixBase<Derived1>& phi,
                                         const Eigen::MatrixBase<Derived2>& J_phi)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 6);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 6, 6);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);

    J.setZero();

    Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
    rightJacobianInvSO3(omega, J.template bottomRightCorner<3, 3>());
    J.template topLeftCorner<3, 3>() = Sophus::SO3<Scalar>::exp(omega).matrix();
}


template <typename Derived1, typename Derived2>
inline void rightJacobianInvDSim3Decoupled(const Eigen::MatrixBase<Derived1>& phi,
                                           const Eigen::MatrixBase<Derived2>& J_phi)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 7);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 7, 7);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);

    J.setZero();

    Eigen::Matrix<Scalar, 3, 1> omega = phi.template segment<3>(3);
    J.template topLeftCorner<3, 3>()  = Sophus::SO3<Scalar>::exp(omega).matrix() * exp(phi(6));
    rightJacobianInvSO3(omega, J.template block<3, 3>(3, 3));
    J(6, 6) = 1;
}



template <typename Scalar>
inline void decoupled_inc(const Sophus::Vector6d& inc, Sophus::SE3<Scalar>& T)
{
    T.translation() += inc.head<3>();
    T.so3() = Sophus::SO3d::exp(inc.tail<3>()) * T.so3();
}

template <typename Scalar>
inline void decoupled_inc(const Sophus::Vector7d& inc, Sophus::DSim3<Scalar>& T)
{
    decoupled_inc(inc.head<6>(), T.se3());
    T.scale() = exp(inc(6)) * T.scale();
}

}  // namespace Sophus



namespace Sophus
{
template <typename T>
inline T translationalError(const Sophus::SE3<T>& a, const Sophus::SE3<T>& b)
{
    Eigen::Matrix<T, 3, 1> diff = a.translation() - b.translation();
    return diff.norm();
}

// the angle (in radian) between two rotations
template <typename T>
inline T rotationalError(const Sophus::SE3<T>& a, const Sophus::SE3<T>& b)
{
    Eigen::Quaternion<T> q1 = a.unit_quaternion();
    Eigen::Quaternion<T> q2 = b.unit_quaternion();
    return q1.angularDistance(q2);
}

// Spherical interpolation
template <typename T>
inline Sophus::SE3<T> slerp(const Sophus::SE3<T>& a, const Sophus::SE3<T>& b, T alpha)
{
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Quat = Eigen::Quaternion<T>;

    Vec3 t  = (1.0 - alpha) * a.translation() + (alpha)*b.translation();
    Quat q1 = a.unit_quaternion();
    Quat q2 = b.unit_quaternion();
    Quat q  = q1.slerp(alpha, q2);
    return Sophus::SE3<T>(q, t);
}

template <typename T>
inline Sophus::SE3<T> mix(const Sophus::SE3<T>& a, const Sophus::SE3<T>& b, T alpha)
{
    return slerp(a, b, alpha);
}

// scale the transformation by a scalar
template <typename T>
inline Sophus::SE3<T> scale(const Sophus::SE3<T>& a, double alpha)
{
    return slerp(Sophus::SE3<T>(), a, alpha);
}

}  // namespace Saiga