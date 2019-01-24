/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/util/assert.h"

#include <Eigen/Core>
#include <Eigen/Geometry>



using quat = Eigen::Quaternionf;

using vec4 = Eigen::Vector4f;
using vec3 = Eigen::Vector3f;
using vec2 = Eigen::Vector2f;

using mat4 = Eigen::Matrix4f;
using mat3 = Eigen::Matrix3f;

using uvec3 = Eigen::Matrix<unsigned int, 3, 1>;

using ivec2 = Eigen::Matrix<int, 2, 1>;
using ivec3 = Eigen::Matrix<int, 3, 1>;
using ivec4 = Eigen::Matrix<int, 4, 1>;

using cvec2 = Eigen::Matrix<char, 2, 1>;
using cvec3 = Eigen::Matrix<char, 3, 1>;
using cvec4 = Eigen::Matrix<char, 4, 1>;

using ucvec2 = Eigen::Matrix<unsigned char, 2, 1>;
using ucvec3 = Eigen::Matrix<unsigned char, 3, 1>;
using ucvec4 = Eigen::Matrix<unsigned char, 4, 1>;


#define IDENTITY_QUATERNION quat::Identity()

template <typename T>
inline T ele_mult(const T& a, const T& b)
{
    return a.array() * b.array();
}

template <typename T>
inline T ele_div(const T& a, const T& b)
{
    return a.array() / b.array();
}

// inline float round(float a)
//{
//    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
//    return a;
//}

inline vec4 round(const vec4& a)
{
    return a.array().round();
}

inline vec3 floor(const vec3& a)
{
    return a.array().floor();
}

inline vec3 operator/(float a, const vec3& b)
{
    return (1.0 / a) * b.array();
}

inline vec3 fract(const vec3& a)
{
    return a.array() - a.array().floor();
}

inline vec3 abs(const vec3& a)
{
    return a.array().abs();
}
inline vec3 clamp(const vec3& x, const vec3& minVal, const vec3& maxVal)
{
    vec3 tmp = x.array().max(minVal.array());
    return tmp.array().max(minVal.array());
}
inline vec4 make_vec4(float x, float y, float z, float w)
{
    return vec4(x, y, z, w);
}

inline vec4 make_vec4(const vec3& v, float a)
{
    return vec4(v(0), v(1), v(2), a);
}
inline vec3 make_vec3(const vec2& v, float a)
{
    return vec3(v(0), v(1), a);
}

inline vec4 make_vec4(float a)
{
    return vec4(a, a, a, a);
}
inline vec3 make_vec3(float a)
{
    return vec3(a, a, a);
}
inline vec3 make_vec3(const vec4& a)
{
    return vec3(a(0), a(1), a(2));
}
inline vec2 make_vec2(float a)
{
    return vec2(a, a);
}
inline vec2 make_vec2(const vec3& a)
{
    return vec2(a(0), a(1));
}

inline ucvec4 make_ucvec4(const ucvec3& v, unsigned char a)
{
    return ucvec4(v(0), v(1), v(2), a);
}



inline vec4 quat_to_vec4(const quat& q)
{
    return q.coeffs();
}


inline float radians(float a)
{
    return a / 180.0 * M_PI;
}

inline mat4 identityMat4()
{
    return mat4::Identity();
}

inline mat4 zeroMat4()
{
    return mat4::Zero();
}


inline mat4 make_mat4(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13, float a20,
                      float a21, float a22, float a23, float a30, float a31, float a32, float a33)
{
    mat4 m;
    m << a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33;
    return m.transpose();
}

inline mat4 make_mat4(const mat3& m)
{
    mat4 m4              = identityMat4();
    m4.block<3, 3>(0, 0) = m;
    return m4;
}

inline mat3 make_mat3(const mat4& m)
{
    return m.block<3, 3>(0, 0);
}

inline mat4 make_mat4(const quat& q)
{
    return make_mat4(q.matrix());
}


inline mat3 make_mat3(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
{
    mat3 m;
    m << a00, a01, a02, a10, a11, a12, a20, a21, a22;
    return m.transpose();
}


inline quat make_quat(float x, float y, float z, float w)
{
    return quat(x, y, z, w);
}


inline quat make_quat(const mat3& m)
{
    return quat(m);
}

inline quat make_quat(const mat4& m)
{
    return make_quat(make_mat3(m));
}


inline quat angleAxis(float angle, const vec3& axis)
{
    Eigen::AngleAxisf aa(angle, axis);
    return make_quat(aa.matrix());
}

inline quat mix(const quat& a, const quat& b, float alpha)
{
    return a.slerp(alpha, b);
}


inline vec4 mix(const vec4& a, const vec4& b, float alpha)
{
    return (1 - alpha) * a + alpha * b;
}

inline vec3 mix(const vec3& a, const vec3& b, float alpha)
{
    return (1 - alpha) * a + alpha * b;
}

inline float mix(const float& a, const float& b, float alpha)
{
    return (1 - alpha) * a + alpha * b;
}

template <typename T>
inline T normalize(const T& v)
{
    return v.normalized();
}

inline vec3 col(const mat3& m, int id)
{
    return m.col(id);
}
inline auto col(mat4& m, int id)
{
    return m.col(id);
}

inline quat quat_cast(const mat3& m)
{
    return quat(m);
}
inline quat quat_cast(const mat4& m)
{
    return quat_cast(make_mat3(m));
}

inline float length(const vec2& v)
{
    return v.norm();
}

inline float length(const vec3& v)
{
    return v.norm();
}

inline quat inverse(const quat& q)
{
    return q.inverse();
}


inline mat4 rotate(float angle, const vec3& axis)
{
    Eigen::AngleAxisf aa(angle, axis);
    return make_mat4(aa.matrix());
}

inline quat rotate(const quat& q, float angle, const vec3& axis)
{
    Eigen::AngleAxisf aa(angle, axis);
    return q * aa;
}
inline mat4 translate(const vec3& t)
{
    mat4 m              = identityMat4();
    m.block<3, 1>(0, 3) = t;
    return m;
}

inline mat4 translate(const mat4& m, const vec3& t)
{
    return m * translate(t);
}

inline mat4 scale(const vec3& t)
{
    mat4 m2  = identityMat4();
    m2(0, 0) = t(0);
    m2(1, 1) = t(1);
    m2(2, 3) = t(2);
    return m2;
}

inline mat4 scale(const mat4& m, const vec3& t)
{
    return m * scale(t);
}

inline mat4 createTRSmatrix(const vec4& t, const quat& r, const vec4& s)
{
    // Equivalent to:
    mat4 T = translate(make_vec3(t));
    mat4 R = make_mat4(r);
    mat4 S = scale(identityMat4(), make_vec3(s));
    return T * R * S;
}



using std::max;
using std::min;

template <typename T>
T clamp(T v, T mi, T ma)
{
    return min(ma, max(mi, v));
}

template <typename T1, typename T2>
inline auto dot(const T1& a, const T2& b)
{
    return a.dot(b);
}

inline vec3 cross(const vec3& a, const vec3& b)
{
    return a.cross(b);
}

inline quat slerp(const quat& a, const quat& b, float alpha)
{
    return a.slerp(alpha, b);
}

inline mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up)
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
inline mat4 perspective(float fovy, float aspect, float zNear, float zFar)
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
inline mat4 ortho(float left, float right, float bottom, float top, float zNear, float zFar)
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

inline float distance(const vec3& a, const vec3& b)
{
    return (a - b).norm();
}

template <typename T>
auto inverse(const T& m)
{
    return m.inverse().eval();
}



template <typename T>
T epsilon()
{
    return std::numeric_limits<T>::epsilon();
}


template <typename T>
T pi()
{
    return T(M_PI);
}

template <typename T>
T two_pi()
{
    return T(M_PI * 2);
}

inline quat rotation(const vec3& a, const vec3& b)
{
    return quat::FromTwoVectors(a, b);
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


namespace Saiga
{
inline std::ostream& operator<<(std::ostream& os, const quat& v)
{
    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
    return os;
}

inline std::istream& operator>>(std::istream& is, vec2& v)
{
    is >> v(0) >> v(1);
    return is;
}
inline std::istream& operator>>(std::istream& is, vec3& v)
{
    is >> v(0) >> v(1) >> v(2);
    return is;
}
inline std::istream& operator>>(std::istream& is, vec4& v)
{
    is >> v(0) >> v(1) >> v(2) >> v(3);
    return is;
}

inline std::string to_string(const mat3& m)
{
    return "mat3";
}
inline std::string to_string(const mat4& m)
{
    return "mat4";
}
inline std::string to_string(const vec3& m)
{
    return "vec3";
}
inline std::string to_string(const vec4& m)
{
    return "vec4";
}

inline mat3 mat3FromString(const std::string& str)
{
    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
    return mat3();
}
inline mat4 mat4FromString(const std::string& str)
{
    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
    return mat4();
}
inline vec3 vec3FromString(const std::string& str)
{
    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
    return vec3();
}
inline vec4 vec4FromString(const std::string& str)
{
    SAIGA_EXIT_ERROR("Use of an unimplemented function!");
    return vec4();
}

}  // namespace Saiga
