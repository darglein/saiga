/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"
#include "saiga/core/util/assert.h"

#include "EigenGLMInterfaceFloat.h"

#include <type_traits>

namespace Saiga
{

#ifdef SAIGA_USE_EIGEN
template <typename Derived>
HD constexpr typename Derived::PlainObject clamp(const Eigen::MatrixBase<Derived>& x,
                                                 const Eigen::MatrixBase<Derived>& minVal,
                                                 const Eigen::MatrixBase<Derived>& maxVal)
{
    typename Derived::PlainObject tmp = x.derived().array().max(minVal.derived().array());
    return tmp.array().min(maxVal.derived().array());
}

template <typename Derived>
HD constexpr typename Derived::PlainObject saturate(const Eigen::MatrixBase<Derived>& x)
{
    typename Derived::PlainObject z, o;
    z.setZero();
    o.setOnes();
    return clamp<Derived>(x, z, o);
}


template <typename Derived1, typename Derived2>
HD constexpr auto mix(const Eigen::MatrixBase<Derived1>& a, const Eigen::MatrixBase<Derived2>& b,
                      typename Derived1::Scalar alpha)
{
    return (1 - alpha) * a + alpha * b;
}

template <typename Derived1, typename Derived2>
HD constexpr auto dot(const Eigen::MatrixBase<Derived1>& a, const Eigen::MatrixBase<Derived2>& b)
{
    return a.dot(b);
}

template <typename Derived1, typename Derived2>
HD constexpr auto cross(const Eigen::MatrixBase<Derived1>& v1, const Eigen::MatrixBase<Derived2>& v2)
{
    return v1.cross(v2);
}

template <typename Derived1, typename Derived2>
HD constexpr auto distance(const Eigen::MatrixBase<Derived1>& v1, const Eigen::MatrixBase<Derived2>& v2)
{
    return (v1 - v2).norm();
}

template <typename Derived1>
HD constexpr auto inverse(const Eigen::MatrixBase<Derived1>& v1)
{
    return v1.inverse();
}

template <typename Derived1>
HD constexpr auto transpose(const Eigen::MatrixBase<Derived1>& v1)
{
    return v1.transpose();
}

template <typename Derived>
HD constexpr typename Derived::Scalar length(const Eigen::MatrixBase<Derived>& v)
{
    return v.norm();
}

// HD constexpr float abs(float v)
//{
//     return std::abs(v);
// }
//
// HD constexpr double abs(double v)
//{
//     return std::abs(v);
// }

using std::abs;

template <typename Derived>
HD constexpr Derived abs(const Eigen::MatrixBase<Derived>& v)
{
    return v.array().abs();
}

template <typename Derived>
HD constexpr Derived normalize(const Eigen::MatrixBase<Derived>& v)
{
    return v.normalized();
}

template <typename Derived>
HD constexpr auto normalize(const Eigen::QuaternionBase<Derived>& q)
{
    return q.normalized();
}


template <typename Derived>
HD constexpr auto slerp(const Eigen::QuaternionBase<Derived>& a, const Eigen::QuaternionBase<Derived>& b,
                        typename Derived::Scalar alpha)
{
    return a.slerp(alpha, b);
}

template <typename Derived>
HD constexpr typename Derived::PlainObject fract(const Eigen::MatrixBase<Derived>& v)
{
    return (v.array() - v.array().floor());
}


//
// Pixar Revised ONB
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
//
// n is aligned to the z-axis
//
template <typename Derived>
Matrix<typename Derived::Scalar, 3, 3> onb(const Eigen::MatrixBase<Derived>& n)
{
    static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, "Input must be 3x1");

    using T    = typename Derived::Scalar;
    using Mat3 = Matrix<T, 3, 3>;
    using Vec3 = Matrix<T, 3, 1>;

    T sign = n(2) > 0 ? 1.0f : -1.0f;  // emulate copysign
    T a    = -1.0f / (sign + n[2]);
    T b    = n[0] * n[1] * a;
    Mat3 v;
    v.col(0) = Vec3(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
    v.col(1) = Vec3(b, sign + n[1] * n[1] * a, -n[1]);
    v.col(2) = n;
    return v;
}

/**
 * Simple ONB from a direction and an up vector.
 */
template <typename Derived1, typename Derived2>
Matrix<typename Derived1::Scalar, 3, 3> onb(const Eigen::MatrixBase<Derived1>& dir,
                                            const Eigen::MatrixBase<Derived2>& up)
{
    using T    = typename Derived1::Scalar;
    using Mat3 = Matrix<T, 3, 3>;

    Mat3 R;
    R.col(2) = dir.normalized();
    R.col(1) = up.normalized();
    R.col(0) = R.col(1).cross(R.col(2)).normalized();
    // make sure it works even if dir and up are not orthogonal
    R.col(1) = R.col(2).cross(R.col(0));
    return R;
}

#else
template <typename _Scalar, int _Rows0, int _Cols0, int _Options0>
HD _Scalar distance(const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& m1,
                    Matrix<_Scalar, _Rows0, _Cols0, _Options0> m2)
{
    return (m1 - m2).norm();
}

template <typename _Scalar, int _Rows0, int _Cols0, int _Options0>
HD Matrix<_Scalar, _Rows0, _Cols0, _Options0> cross(const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& m1,
                                                           Matrix<_Scalar, _Rows0, _Cols0, _Options0> m2)
{
    return m1.cross(m2);
}

template <typename _Scalar, int _Rows0, int _Cols0, int _Options0>
HD _Scalar dot(const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& m1,
               Matrix<_Scalar, _Rows0, _Cols0, _Options0> m2)
{
    return m1.dot(m2);
}


template <typename _Scalar, int _Rows0, int _Cols0, int _Options0>
HD Matrix<_Scalar, _Rows0, _Cols0, _Options0> normalize(
    const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& m1)
{
    return m1.normalized();
}

template <typename _Scalar, int _Rows0, int _Cols0, int _Options0>
HD Matrix<_Scalar, _Rows0, _Cols0, _Options0> ArrayMult(
    const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& m1, Matrix<_Scalar, _Rows0, _Cols0, _Options0> m2)
{
    return m1.array() * m2.array();
}


template <typename _Scalar, int _Rows0, int _Cols0, int _Options0>
HD Matrix<_Scalar, _Rows0, _Cols0, _Options0> inverse(
    const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& m1)
{
    return m1.inverse();
}


template <typename _Scalar, int _Rows0, int _Cols0, int _Options0>
HD constexpr auto mix(const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& a,
                      const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& b, _Scalar alpha)
{
    return (1 - alpha) * a + alpha * b;
}

template <typename _Scalar, int _Rows0, int _Cols0, int _Options0>
HD _Scalar length(const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& m1)
{
    return m1.norm();
}

template <typename Scalar>
HD constexpr auto slerp(const TinyEigen::Quaternion<Scalar>& a, const TinyEigen::Quaternion<Scalar>& b, Scalar alpha)
{
    return a.slerp(alpha, b);
}

template <typename Scalar>
HD constexpr auto normalize(const TinyEigen::Quaternion<Scalar>& a)
{
    return a.normalized();
}


template <typename Derived>
HD constexpr Derived abs(const TinyEigen::MatrixBase<Derived>& v)
{
    return v.derived().array().abs();
}

template <typename Derived>
HD constexpr typename Derived::PlainObject clamp(const TinyEigen::MatrixBase<Derived>& x,
                                                 const TinyEigen::MatrixBase<Derived>& minVal,
                                                 const TinyEigen::MatrixBase<Derived>& maxVal)
{
    typename Derived::PlainObject tmp = x.derived().array().max(minVal.derived().array());
    return tmp.array().min(maxVal.derived().array());
}

template <typename Derived>
HD constexpr typename Derived::PlainObject fract(const TinyEigen::MatrixBase<Derived>& v)
{
    return (v.derived().array() - v.derived().array().floor());
}



template <typename Derived>
HD constexpr typename Derived::PlainObject saturate(const TinyEigen::MatrixBase<Derived>& x)
{
    typename Derived::PlainObject z, o;
    z.setZero();
    o.setOnes();
    return clamp<Derived>(x, z, o);
}


//
// Pixar Revised ONB
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
//
// n is aligned to the z-axis
//
template <typename Derived>
Matrix<typename Derived::Scalar, 3, 3> onb(const TinyEigen::MatrixBase<Derived>& n)
{
    using T    = typename Derived::Scalar;
    using Mat3 = Matrix<T, 3, 3>;
    using Vec3 = Matrix<T, 3, 1>;

    T sign = n(2) > 0 ? 1.0f : -1.0f;  // emulate copysign
    T a    = -1.0f / (sign + n(2));
    T b    = n(0) * n(1) * a;
    Mat3 v;
    v.col(0) = Vec3(1.0f + sign * n(0) * n(0) * a, sign * b, -sign * n(0));
    v.col(1) = Vec3(b, sign + n(1) * n(1) * a, -n(1));
    v.col(2) = n;
    return v;
}


using std::abs;

#endif

SAIGA_CORE_API extern mat4 scale(const vec3& t);
SAIGA_CORE_API extern mat4 translate(const vec3& t);
SAIGA_CORE_API extern mat4 rotate(float angle, const vec3& axis);
SAIGA_CORE_API extern quat rotate(const quat& q, float angle, const vec3& axis);

SAIGA_CORE_API extern quat angleAxis(float angle, const vec3& axis);
SAIGA_CORE_API extern Quat angleAxis(double angle, const Vec3& axis);
SAIGA_CORE_API extern quat mix(const quat& a, const quat& b, float alpha);
SAIGA_CORE_API extern Quat mix(const Quat& a, const Quat& b, double alpha);
SAIGA_CORE_API extern quat quat_cast(const mat3& m);
SAIGA_CORE_API extern quat quat_cast(const mat4& m);
SAIGA_CORE_API extern quat inverse(const quat& q);

SAIGA_CORE_API extern quat rotation(const vec3& a, const vec3& b);

SAIGA_CORE_API extern vec4 make_vec4(float x, float y, float z, float w);
SAIGA_CORE_API extern vec4 make_vec4(const vec3& v, float a);
SAIGA_CORE_API extern vec4 make_vec4(const vec2& v, const vec2& v2);
SAIGA_CORE_API extern vec4 make_vec4(float a);

SAIGA_CORE_API extern vec3 make_vec3(const vec2& v, float a);
SAIGA_CORE_API extern vec3 make_vec3(float a);
SAIGA_CORE_API extern vec3 make_vec3(const vec4& a);

SAIGA_CORE_API extern vec2 make_vec2(float a);
SAIGA_CORE_API extern vec2 make_vec2(const vec3& a);
SAIGA_CORE_API extern vec2 make_vec2(float a, float b);
SAIGA_CORE_API extern vec2 make_vec2(const ivec2& a);

SAIGA_CORE_API extern ivec2 make_ivec2(int a, int b);
SAIGA_CORE_API extern ucvec4 make_ucvec4(const ucvec3& v, unsigned char a);
SAIGA_CORE_API extern mat4 make_mat4(float a00, float a01, float a02, float a03, float a10, float a11, float a12,
                                     float a13, float a20, float a21, float a22, float a23, float a30, float a31,
                                     float a32, float a33);
SAIGA_CORE_API extern mat4 make_mat4_row_major(float a00, float a01, float a02, float a03, float a10, float a11,
                                               float a12, float a13, float a20, float a21, float a22, float a23,
                                               float a30, float a31, float a32, float a33);
SAIGA_CORE_API extern mat4 make_mat4(const mat3& m);
SAIGA_CORE_API extern mat3 make_mat3(const mat4& m);
SAIGA_CORE_API extern mat3 make_mat3(float a00, float a01, float a02, float a10, float a11, float a12, float a20,
                                     float a21, float a22);

SAIGA_CORE_API extern mat4 make_mat4(const quat& q);
SAIGA_CORE_API extern mat3 make_mat3(const quat& q);
SAIGA_CORE_API extern quat make_quat(float x, float y, float z, float w);
SAIGA_CORE_API extern vec4 quat_to_vec4(const quat& q);
SAIGA_CORE_API extern quat make_quat(const mat3& m);
SAIGA_CORE_API extern quat make_quat(const mat4& m);

SAIGA_CORE_API extern mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up);
SAIGA_CORE_API extern mat4 perspective(float fovy, float aspect, float zNear, float zFar);
SAIGA_CORE_API extern mat4 ortho(float left, float right, float bottom, float top, float zNear, float zFar);

SAIGA_CORE_API extern mat4 createTRSmatrix(const vec3& t, const quat& r, const vec3& s);
SAIGA_CORE_API extern mat4 createTRSmatrix(const vec4& t, const quat& r, const vec4& s);

// ===== Double overloads (R-MATH-2 of docs/engine_design.md) =====
// Templates restricted to double scalars, so that float arguments keep using the float functions above and Eigen
// expressions (e.g. eye + dir) are not ambiguous between the float and the double version.
namespace detail
{
template <typename... Derived>
constexpr bool AllDouble = (std::is_same_v<typename Derived::Scalar, double> && ...);
// A 3D or 4D vector
template <typename D>
constexpr bool Vec3or4 = (D::RowsAtCompileTime == 3 || D::RowsAtCompileTime == 4) && D::ColsAtCompileTime == 1;
// A 3x3 or 4x4 matrix
template <typename D>
constexpr bool Mat3or4 =
    (D::RowsAtCompileTime == 3 || D::RowsAtCompileTime == 4) && D::ColsAtCompileTime == D::RowsAtCompileTime;
}  // namespace detail

// Translation * rotation * scale
template <typename D1, typename D2, std::enable_if_t<detail::AllDouble<D1, D2>, int> = 0>
Mat4 createTRSmatrix(const MatrixBase<D1>& t, const Quat& r, const MatrixBase<D2>& s)
{
    static_assert(detail::Vec3or4<D1> && detail::Vec3or4<D2>, "t and s must be 3D or 4D vectors");
    Mat4 m = Mat4::Identity();
    Mat3 R = r.matrix();
    for (int i = 0; i < 3; ++i)
    {
        // Two-index access: block views (MatrixView) have no single-index operator
        for (int j = 0; j < 3; ++j) m(i, j) = R(i, j) * s(j, 0);
        m(i, 3) = t(i, 0);
    }
    return m;
}

// View matrix of a right handed camera at eye looking at center (like the float lookAt). Requires eye != center and
// up not parallel to the view direction (asserted; like the float version the result is not usable otherwise).
template <typename D1, typename D2, typename D3, std::enable_if_t<detail::AllDouble<D1, D2, D3>, int> = 0>
Mat4 lookAt(const MatrixBase<D1>& eye, const MatrixBase<D2>& center, const MatrixBase<D3>& up)
{
    static_assert(D1::RowsAtCompileTime == 3 && D2::RowsAtCompileTime == 3 && D3::RowsAtCompileTime == 3 &&
                      D1::ColsAtCompileTime == 1 && D2::ColsAtCompileTime == 1 && D3::ColsAtCompileTime == 1,
                  "lookAt takes 3D vectors");
    Vec3 e = eye, c = center, u0 = up;
    SAIGA_ASSERT((c - e).squaredNorm() > 0, "lookAt: eye == center");
    Vec3 f = (c - e).normalized();
    SAIGA_ASSERT(f.cross(u0).squaredNorm() > 0, "lookAt: up is parallel to the view direction");
    Vec3 s = f.cross(u0).normalized();
    Vec3 u = s.cross(f);

    Mat4 m  = Mat4::Identity();
    m(0, 0) = s.x();
    m(0, 1) = s.y();
    m(0, 2) = s.z();
    m(1, 0) = u.x();
    m(1, 1) = u.y();
    m(1, 2) = u.z();
    m(2, 0) = -f.x();
    m(2, 1) = -f.y();
    m(2, 2) = -f.z();
    m(0, 3) = -s.dot(e);
    m(1, 3) = -u.dot(e);
    m(2, 3) = f.dot(e);
    return m;
}

// Rotation of a 3x3 or 4x4 (upper left 3x3) rotation matrix. make_quat does not normalize, quat_cast does (like the
// float versions).
template <typename D, std::enable_if_t<detail::AllDouble<D>, int> = 0>
Quat make_quat(const MatrixBase<D>& m)
{
    static_assert(detail::Mat3or4<D>, "make_quat takes a 3x3 or 4x4 matrix");
    Mat3 R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R(i, j) = m(i, j);
    return Quat(R);
}

template <typename D, std::enable_if_t<detail::AllDouble<D>, int> = 0>
Quat quat_cast(const MatrixBase<D>& m)
{
    return make_quat(m).normalized();
}

}  // namespace Saiga
