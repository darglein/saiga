/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <Eigen/Core>
#include <Eigen/Geometry>



namespace Saiga {



using quat = Eigen::Quaternionf;

using vec4 = Eigen::Vector4f;
using vec3 = Eigen::Vector3f;
using vec2 = Eigen::Vector2f;

using mat4 = Eigen::Matrix4f;
using mat3 = Eigen::Matrix3f;

using uvec3 = Eigen::Matrix<unsigned int, 3,1>;

using ivec2 = Eigen::Matrix<int, 2,1>;
using ivec3 = Eigen::Matrix<int, 3,1>;
using ivec4 = Eigen::Matrix<int, 4,1>;

using cvec2 = Eigen::Matrix<char, 2,1>;
using cvec3 = Eigen::Matrix<char, 3,1>;
using cvec4 = Eigen::Matrix<char, 4,1>;

using ucvec2 = Eigen::Matrix<unsigned char, 2,1>;
using ucvec3 = Eigen::Matrix<unsigned char, 3,1>;
using ucvec4 = Eigen::Matrix<unsigned char, 4,1>;


#define IDENTITY_QUATERNION quat::Identity()

inline mat4 make_mat4(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13, float a20, float a21, float a22, float a23, float a30, float a31, float a32, float a33)
{
    return mat4();
}

inline float radians(float a)
{
    return a / 180.0 * M_PI;
}

inline mat4 identityMat4() { return mat4::Identity(); }

inline vec4 mix(const vec4& a, const vec4& b, float alpha) { return (1-alpha) * a + alpha * b; }

inline vec3 normalize(const vec3& v ) { return v.normalized();}
inline quat normalize(const quat& v ) { return v.normalized();}

inline vec3 col(const mat3& m, int id) { return m.col(id); }
inline vec4 col(const mat4& m, int id) { return m.col(id); }

inline quat quat_cast(const mat3& m) { return quat(); }
inline float length(const vec3& v) { return v.norm(); }
inline mat4 inverse(const mat4& m) { return m.inverse(); }
inline mat3 inverse(const mat3& m) { return m.inverse(); }

inline quat inverse(const quat& q) { return q.inverse(); }

inline quat rotate(const quat& q, float angle, vec3 axis) { return q; }
inline mat4 translate(const mat4& m, vec3 t) { return m; }
inline mat4 scale(const mat4& m, vec3 t) { return m; }

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

template<typename T>
T clamp(T v, T mi, T ma)
{
    return min(ma,max(mi,v));
}


inline float dot(vec3 a, vec3 b)
{
    return a.dot(b);
}

inline vec3 cross(vec3 a, vec3 b){return a.cross(b);}

inline quat slerp(const quat& a, const quat& b, float alpha) { return a.slerp(alpha,b); }


}
