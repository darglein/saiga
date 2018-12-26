/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

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

namespace Saiga
{
#define IDENTITY_QUATERNION quat::Identity()

inline vec3 ele_mult(vec3 a, vec3 b)
{
    return a;
}

inline vec4 round(vec4 a)
{
    return a;
}

inline vec3 fract(vec3 a)
{
    return a;
}
inline float abs(float a)
{
    return a;
}
inline vec3 abs(vec3 a)
{
    return a;
}
inline vec3 clamp(vec3 a, vec3 b, vec3 c)
{
    return a;
}
inline vec4 make_vec4(float x, float y, float z, float w)
{
    return vec4();
}

inline vec4 make_vec4(const vec3& v, float a)
{
    return vec4();
}
inline vec3 make_vec3(const vec2& v, float a)
{
    return vec3();
}

inline vec4 make_vec4(float a)
{
    return vec4();
}
inline vec3 make_vec3(float a)
{
    return vec3();
}
inline vec3 make_vec3(vec4 a)
{
    return vec3();
}
inline vec2 make_vec2(float a)
{
    return vec2();
}
inline vec2 make_vec2(vec3 a)
{
    return vec2();
}

inline ucvec4 make_ucvec4(const ucvec3& v, unsigned char a)
{
    return ucvec4();
}

inline quat make_quat(float x, float y, float z, float w)
{
    return quat(x, y, z, w);
}

inline quat make_quat(const mat4& m)
{
    return IDENTITY_QUATERNION;
}

inline quat angleAxis(float angle, vec3 axis)
{
    return quat();
}

inline vec4 quat_to_vec4(quat q)
{
    return q.coeffs();
}

inline mat4 make_mat4(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13, float a20,
                      float a21, float a22, float a23, float a30, float a31, float a32, float a33)
{
    return mat4();
}

inline mat3 make_mat3(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13, float a20)
{
    return mat3();
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
    //    return m.block(0, 0, 4, 1);
}

inline quat quat_cast(const mat3& m)
{
    return quat();
}
inline quat quat_cast(const mat4& m)
{
    return quat();
}

inline float length(const vec3& v)
{
    return v.norm();
}

inline quat inverse(const quat& q)
{
    return q.inverse();
}

inline quat rotate(const quat& q, float angle, vec3 axis)
{
    return q;
}
inline mat4 translate(const mat4& m, vec3 t)
{
    return m;
}
inline mat4 scale(const mat4& m, vec3 t)
{
    return m;
}

SAIGA_GLOBAL inline mat4 createTRSmatrix(const vec4& t, const quat& r, const vec4& s)
{
    // Equivalent to:
    //    mat4 T = translate(mat4(1),vec3(t));
    //    mat4 R = mat4_cast(r);
    //    mat4 S = scale(mat4(1),vec3(s));
    //    return T * R * S;


    return mat4::Identity();
}



using std::max;
using std::min;

template <typename T>
T clamp(T v, T mi, T ma)
{
    return min(ma, max(mi, v));
}

template <typename T1, typename T2>
inline auto dot(T1 a, T2 b)
{
    return a.dot(b);
}

inline vec3 cross(vec3 a, vec3 b)
{
    return a.cross(b);
}

inline quat slerp(const quat& a, const quat& b, float alpha)
{
    return a.slerp(alpha, b);
}

inline mat4 lookAt(vec3 eye, vec3 center, vec3 up)
{
    return mat4();
}
inline mat4 perspective(float a, float b, float c, float d)
{
    return mat4();
}
inline mat4 ortho(float a, float b, float c, float d, float e, float f)
{
    return mat4();
}

inline float distance(vec3 a, vec3 b)
{
    return 0;
}

template <typename T>
auto inverse(const T& m)
{
    return m.inverse().eval();
}



inline std::ostream& operator<<(std::ostream& os, const quat& v)
{
    return os;
}

inline std::istream& operator>>(std::istream& is, vec2& v)
{
    return is;
}
inline std::istream& operator>>(std::istream& is, vec3& v)
{
    return is;
}
inline std::istream& operator>>(std::istream& is, vec4& v)
{
    return is;
}

inline float linearRand(float low, float high)
{
    return 0;
}

template <typename T>
T epsilon()
{
    return T(0);
}

}  // namespace Saiga
