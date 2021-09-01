/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"

#include "EigenGLMInterfaceFloat.h"

namespace Saiga
{
template <typename Derived>
HD constexpr typename Derived::PlainObject clamp(const Eigen::EigenBase<Derived>& x,
                                                 const Eigen::EigenBase<Derived>& minVal,
                                                 const Eigen::EigenBase<Derived>& maxVal)
{
    typename Derived::PlainObject tmp = x.derived().array().max(minVal.derived().array());
    return tmp.array().min(maxVal.derived().array());
}

template <typename Derived>
HD constexpr typename Derived::PlainObject saturate(const Eigen::EigenBase<Derived>& x)
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

//HD constexpr float abs(float v)
//{
//    return std::abs(v);
//}
//
//HD constexpr double abs(double v)
//{
//    return std::abs(v);
//}

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
    v.col(2) = n;
    v.col(1) = Vec3(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
    v.col(0) = Vec3(b, sign + n[1] * n[1] * a, -n[1]);
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



SAIGA_CORE_API extern mat4 scale(const vec3& t);
SAIGA_CORE_API extern mat4 translate(const vec3& t);
SAIGA_CORE_API extern mat4 rotate(float angle, const vec3& axis);
SAIGA_CORE_API extern quat rotate(const quat& q, float angle, const vec3& axis);

SAIGA_CORE_API extern quat angleAxis(float angle, const vec3& axis);
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


}  // namespace Saiga
