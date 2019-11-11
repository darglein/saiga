/**
 * Copyright (c) 2017 Darius Rückert
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

SAIGA_CORE_API extern mat4 scale(const vec3& t);
SAIGA_CORE_API extern mat4 translate(const vec3& t);
SAIGA_CORE_API extern mat4 rotate(float angle, const vec3& axis);
SAIGA_CORE_API extern quat rotate(const quat& q, float angle, const vec3& axis);

SAIGA_CORE_API extern quat angleAxis(float angle, const vec3& axis);
SAIGA_CORE_API extern quat mix(const quat& a, const quat& b, float alpha);
SAIGA_CORE_API extern quat quat_cast(const mat3& m);
SAIGA_CORE_API extern quat quat_cast(const mat4& m);
SAIGA_CORE_API extern quat inverse(const quat& q);

SAIGA_CORE_API extern quat slerp(const quat& a, const quat& b, float alpha);
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
SAIGA_CORE_API extern mat4 createTRSmatrix(const vec4& t, const quat& r, const vec4& s);


}  // namespace Saiga
