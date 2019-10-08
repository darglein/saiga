/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"

namespace Saiga
{
using std::abs;
using std::max;
using std::min;
using std::round;



template <typename T>
constexpr T epsilon()
{
    return std::numeric_limits<T>::epsilon();
}


template <typename T>
constexpr T pi()
{
    return T(3.14159265358979323846);
}

template <typename T>
constexpr T two_pi()
{
    return pi<T>() * T(2);
}


template <typename T>
HD inline T ele_mult(const T& a, const T& b)
{
    return a.array() * b.array();
}

template <typename T>
HD inline T ele_div(const T& a, const T& b)
{
    return a.array() / b.array();
}

HD inline vec4 min(const vec4& a, const vec4& b)
{
    return a.array().min(b.array());
}

HD inline vec4 max(const vec4& a, const vec4& b)
{
    return a.array().max(b.array());
}

HD inline vec3 min(const vec3& a, const vec3& b)
{
    return a.array().min(b.array());
}

HD inline vec3 max(const vec3& a, const vec3& b)
{
    return a.array().max(b.array());
}

HD inline vec2 min(const vec2& a, const vec2& b)
{
    return a.array().min(b.array());
}

HD inline vec2 max(const vec2& a, const vec2& b)
{
    return a.array().max(b.array());
}

HD inline vec4 round(const vec4& a)
{
    return a.array().round();
}

HD inline vec3 floor(const vec3& a)
{
    return a.array().floor();
}

HD inline vec3 operator/(float a, const vec3& b)
{
    return (1.0 / a) * b.array();
}

HD inline float fract(float a)
{
    return a - std::floor(a);
}

HD inline vec3 fract(const vec3& a)
{
    return a.array() - a.array().floor();
}

HD inline vec2 abs(const vec2& a)
{
    return a.array().abs();
}

HD inline vec3 abs(const vec3& a)
{
    return a.array().abs();
}
HD inline vec3 clamp(const vec3& x, const vec3& minVal, const vec3& maxVal)
{
    vec3 tmp = x.array().max(minVal.array());
    return tmp.array().min(maxVal.array());
}

HD inline ivec2 clamp(const ivec2& x, const ivec2& minVal, const ivec2& maxVal)
{
    ivec2 tmp = x.array().max(minVal.array());
    return tmp.array().min(maxVal.array());
}

HD inline vec4 make_vec4(float x, float y, float z, float w)
{
    return vec4(x, y, z, w);
}

HD inline vec4 make_vec4(const vec3& v, float a)
{
    return vec4(v(0), v(1), v(2), a);
}
HD inline vec3 make_vec3(const vec2& v, float a)
{
    return vec3(v(0), v(1), a);
}

HD inline vec4 make_vec4(const vec2& v, const vec2& v2)
{
    return vec4(v(0), v(1), v2(0), v2(1));
}

HD inline vec4 make_vec4(float a)
{
    return vec4(a, a, a, a);
}
HD inline vec3 make_vec3(float a)
{
    return vec3(a, a, a);
}
HD inline vec3 make_vec3(const vec4& a)
{
    return vec3(a(0), a(1), a(2));
}
HD inline vec2 make_vec2(float a)
{
    return vec2(a, a);
}
HD inline vec2 make_vec2(const vec3& a)
{
    return vec2(a(0), a(1));
}

HD inline vec2 make_vec2(float a, float b)
{
    return vec2(a, b);
}

HD inline vec2 make_vec2(const ivec2& a)
{
    return a.cast<float>();
}

HD inline ivec2 make_ivec2(int a, int b)
{
    ivec2 v;
    v(0) = a;
    v(1) = b;
    return v;
}


HD inline ucvec4 make_ucvec4(const ucvec3& v, unsigned char a)
{
    return ucvec4(v(0), v(1), v(2), a);
}

HD inline float degrees(float a)
{
    return a * 180.0 / pi<float>();
}

HD inline float radians(float a)
{
    return a / 180.0 * pi<float>();
}

HD inline mat3 identityMat3()
{
    return mat3::Identity();
}

HD inline mat4 identityMat4()
{
    return mat4::Identity();
}

HD inline mat4 zeroMat4()
{
    return mat4::Zero();
}


HD inline mat4 make_mat4(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13,
                         float a20, float a21, float a22, float a23, float a30, float a31, float a32, float a33)
{
    mat4 m;
    m << a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33;
    return m.transpose();
}

HD inline mat4 make_mat4_row_major(float a00, float a01, float a02, float a03, float a10, float a11, float a12,
                                   float a13, float a20, float a21, float a22, float a23, float a30, float a31,
                                   float a32, float a33)
{
    mat4 m;
    m << a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33;
    return m;
}

HD inline mat4 make_mat4(const mat3& m)
{
    mat4 m4              = identityMat4();
    m4.block<3, 3>(0, 0) = m;
    return m4;
}

HD inline mat3 make_mat3(const mat4& m)
{
    return m.block<3, 3>(0, 0);
}



HD inline mat3 make_mat3(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21,
                         float a22)
{
    mat3 m;
    m << a00, a01, a02, a10, a11, a12, a20, a21, a22;
    return m.transpose();
}



HD inline vec4 mix(const vec4& a, const vec4& b, float alpha)
{
    return (1 - alpha) * a + alpha * b;
}

HD inline vec3 mix(const vec3& a, const vec3& b, float alpha)
{
    return (1 - alpha) * a + alpha * b;
}

HD inline vec2 mix(const vec2& a, const vec2& b, float alpha)
{
    return (1 - alpha) * a + alpha * b;
}

HD inline float mix(const float& a, const float& b, float alpha)
{
    return (1 - alpha) * a + alpha * b;
}



template <typename T>
HD inline T normalize(const T& v)
{
    return v.normalized();
}

HD inline auto col(const mat3& m, int id)
{
    return m.col(id);
}

HD inline auto col(mat3& m, int id)
{
    return m.col(id);
}

HD inline auto col(mat4& m, int id)
{
    return m.col(id);
}

HD inline auto col(const mat4& m, int id)
{
    return m.col(id);
}



HD inline float length(const vec2& v)
{
    return v.norm();
}

HD inline float length(const vec3& v)
{
    return v.norm();
}


HD inline mat4 translate(const vec3& t)
{
    mat4 m              = identityMat4();
    m.block<3, 1>(0, 3) = t;
    return m;
}

HD inline mat4 translate(const mat4& m, const vec3& t)
{
    return m * translate(t);
}

HD inline mat4 scale(const vec3& t)
{
    mat4 m2  = identityMat4();
    m2(0, 0) = t(0);
    m2(1, 1) = t(1);
    m2(2, 2) = t(2);
    return m2;
}

HD inline mat4 scale(const mat4& m, const vec3& t)
{
    return m * scale(t);
}



template <typename T>
constexpr T clamp(T v, T mi, T ma)
{
    return min(ma, max(mi, v));
}

template <typename T1, typename T2>
constexpr HD inline auto dot(const T1& a, const T2& b)
{
    return a.dot(b);
}

HD inline vec3 cross(const vec3& a, const vec3& b)
{
    return a.cross(b);
}


HD inline mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up)
{
    // right handed coordinate system

    vec3 const f(normalize((center - eye).eval()));
    vec3 const s(normalize(cross(f, up)));
    vec3 const u(cross(s, f));

    mat4 Result  = identityMat4();
    Result(0, 0) = s.x();
    Result(0, 1) = s.y();
    Result(0, 2) = s.z();
    Result(1, 0) = u.x();
    Result(1, 1) = u.y();
    Result(1, 2) = u.z();
    Result(2, 0) = -f.x();
    Result(2, 1) = -f.y();
    Result(2, 2) = -f.z();
    Result(0, 3) = -dot(s, eye);
    Result(1, 3) = -dot(u, eye);
    Result(2, 3) = dot(f, eye);
    return Result;
}
HD inline mat4 perspective(float fovy, float aspect, float zNear, float zFar)
{
    float const tanHalfFovy = tan(fovy / static_cast<float>(2));

    mat4 Result  = zeroMat4();
    Result(0, 0) = static_cast<float>(1) / (aspect * tanHalfFovy);
    Result(1, 1) = static_cast<float>(1) / (tanHalfFovy);
    Result(2, 2) = -(zFar + zNear) / (zFar - zNear);
    Result(3, 2) = -static_cast<float>(1);
    Result(2, 3) = -(static_cast<float>(2) * zFar * zNear) / (zFar - zNear);
    return Result;
}
HD inline mat4 ortho(float left, float right, float bottom, float top, float zNear, float zFar)
{
    mat4 Result  = identityMat4();
    Result(0, 0) = static_cast<float>(2) / (right - left);
    Result(1, 1) = static_cast<float>(2) / (top - bottom);
    Result(2, 2) = -static_cast<float>(2) / (zFar - zNear);
    Result(0, 3) = -(right + left) / (right - left);
    Result(1, 3) = -(top + bottom) / (top - bottom);
    Result(2, 3) = -(zFar + zNear) / (zFar - zNear);
    return Result;
}

HD inline float distance(const vec3& a, const vec3& b)
{
    return (a - b).norm();
}


HD inline float distance(const vec2& a, const vec2& b)
{
    return (a - b).norm();
}


HD inline auto inverse(const mat3& m)
{
    return m.inverse().eval();
}

HD inline auto inverse(const mat4& m)
{
    return m.inverse().eval();
}

HD inline auto transpose(const mat3& m)
{
    return m.transpose().eval();
}

HD inline auto transpose(const mat4& m)
{
    return m.transpose().eval();
}



template <typename _Scalar, int _Rows, int _Cols>
_Scalar* data(Eigen::Matrix<_Scalar, _Rows, _Cols>& M)
{
    return M.data();
}

template <typename _Scalar, int _Rows, int _Cols>
const _Scalar* data(const Eigen::Matrix<_Scalar, _Rows, _Cols>& M)
{
    return M.data();
}



}  // namespace Saiga

namespace Saiga
{
// HD inline std::string to_string(const mat3& m)
//{
//    //    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
//    return "mat3";
//}
// HD inline std::string to_string(const mat4& m)
//{
//    //    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
//    return "mat4";
//}
// HD inline std::string to_string(const vec3& m)
//{
//    //    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
//    return "vec3";
//}
// HD inline std::string to_string(const vec4& m)
//{
//    //    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
//    return "vec4";
//}

// HD inline mat3 mat3FromString(const std::string& str)
//{
//    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
//    return mat3();
//}
// HD inline mat4 mat4FromString(const std::string& str)
//{
//    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
//    return mat4();
//}
// HD inline vec3 vec3FromString(const std::string& str)
//{
//    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
//    return vec3();
//}
// HD inline vec4 vec4FromString(const std::string& str)
//{
//    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
//    return vec4();
//}

}  // namespace Saiga


namespace Saiga
{
HD inline mat4 make_mat4(const quat& q)
{
    return make_mat4(q.matrix());
}

HD inline mat3 make_mat3(const quat& q)
{
    return q.matrix();
}

HD inline quat make_quat(float x, float y, float z, float w)
{
    // Eigen quats are stored as (x,y,z,w), but the constructor is (w,x,y,z)
    return quat(w, x, y, z);
}

// return vec4 as (x,y,z,w)
HD inline vec4 quat_to_vec4(const quat& q)
{
    return vec4(q.x(), q.y(), q.z(), q.w());
}


HD inline quat make_quat(const mat3& m)
{
    return quat(m);
}

HD inline quat make_quat(const mat4& m)
{
    return make_quat(make_mat3(m));
}


HD inline quat angleAxis(float angle, const vec3& axis)
{
    Eigen::AngleAxisf aa(angle, axis);
    return make_quat(aa.matrix());
}

HD inline quat mix(const quat& a, const quat& b, float alpha)
{
    return a.slerp(alpha, b);
}
HD inline quat quat_cast(const mat3& m)
{
    return quat(m).normalized();
}
HD inline quat quat_cast(const mat4& m)
{
    return quat_cast(make_mat3(m));
}

HD inline quat inverse(const quat& q)
{
    return q.inverse();
}



HD inline mat4 rotate(float angle, const vec3& axis)
{
    Eigen::AngleAxisf aa(angle, axis);
    return make_mat4(aa.matrix());
}

HD inline quat rotate(const quat& q, float angle, const vec3& axis)
{
    Eigen::AngleAxisf aa(angle, axis);
    return q * aa;
}
HD inline mat4 rotate(const mat4& m, float angle, const vec3& axis)
{
    return m * rotate(angle, axis);
}
HD inline quat slerp(const quat& a, const quat& b, float alpha)
{
    return a.slerp(alpha, b);
}

HD inline quat rotation(const vec3& a, const vec3& b)
{
    return quat::FromTwoVectors(a, b);
}

HD inline float smoothstep(float edge0, float edge1, float x)
{
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

HD inline mat4 createTRSmatrix(const vec4& t, const quat& r, const vec4& s)
{
    // Equivalent to:
    mat4 T = translate(make_vec3(t));
    mat4 R = make_mat4(r);
    mat4 S = scale(identityMat4(), make_vec3(s));
    return T * R * S;
}



}  // namespace Saiga



#define IDENTITY_QUATERNION Saiga::quat::Identity()
