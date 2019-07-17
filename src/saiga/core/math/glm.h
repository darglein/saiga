/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#ifndef SAIGA_USE_GLM
#error Saiga was compiled without glm.
#endif

#include <iostream>
#include <string>
//#define GLM_USE_SSE

#define GLM_FORCE_RADIANS

#if defined(GLM_USE_SSE)
//#define GLM_FORCE_ALIGNED
#    define GLM_SIMD_ENABLE_XYZW_UNION
#endif

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtx/type_aligned.hpp>


#if defined(GLM_USE_SSE)


//#include <glm/gtx/simd_quat.hpp>
//#include <glm/gtx/simd_vec4.hpp>
//#include <glm/gtx/simd_mat4.hpp>

#    if GLM_VERSION != 98
#        error GLM Version not supported!
#    endif

// In version 98 this part is missing in glm/gtc/random.inl
namespace glm
{
namespace detail
{
template <template <class, precision> class vecType>
struct compute_linearRand<float, aligned_highp, vecType>
{
    GLM_FUNC_QUALIFIER static vecType<float, aligned_highp> call(vecType<float, aligned_highp> const& Min,
                                                                 vecType<float, aligned_highp> const& Max)
    {
        return vecType<float, aligned_highp>(compute_rand<uint32, aligned_highp, vecType>::call()) /
                   static_cast<float>(std::numeric_limits<uint32>::max()) * (Max - Min) +
               Min;
    }
};
}  // namespace detail
}  // namespace glm

typedef tvec2<float, precision::aligned_highp> vec2;
typedef tvec3<float, precision::aligned_highp> vec3;
typedef tvec4<float, precision::aligned_highp> avec4;
typedef tmat4x4<float, precision::aligned_highp> amat4;
typedef tquat<float, precision::aligned_highp> aquat;

// GLM_ALIGNED_TYPEDEF(avec2, vec2, 16);
// GLM_ALIGNED_TYPEDEF(avec3, vec3, 16);
GLM_ALIGNED_TYPEDEF(avec4, vec4, 16);
GLM_ALIGNED_TYPEDEF(amat4, mat4, 16);
GLM_ALIGNED_TYPEDEF(aquat, quat, 16);


// GLM_ALIGNED_TYPEDEF(vec2, vec2, 16);
// GLM_ALIGNED_TYPEDEF(vec3, vec3, 16);
// typedef vec2 vec2;
// typedef vec3 vec3;
// typedef simdVec4 vec4;
// typedef simdQuat quat;
// typedef simdMat4 mat4;

// GLM_ALIGNED_TYPEDEF(vec4, vec4, 16);
// GLM_ALIGNED_TYPEDEF(mat4, mat4, 16);
// GLM_ALIGNED_TYPEDEF(quat, quat, 16);

#else

namespace Saiga
{
using glm::ivec2;
using glm::uvec3;

using glm::vec2;
using glm::vec3;
using glm::vec4;

using glm::mat3;
using glm::mat4;

using glm::quat;

using glm::clamp;
using glm::degrees;
using glm::max;
using glm::min;
using glm::mix;
using glm::ortho;
using glm::perspective;
using glm::radians;
using glm::scale;
using glm::smoothstep;

// random
using glm::diskRand;
using glm::linearRand;
using glm::sphericalRand;

using glm::epsilon;
using glm::pi;
using glm::two_pi;



#    define IDENTITY_QUATERNION quat(1, 0, 0, 0)

using cvec2 = glm::tvec2<char, glm::highp>;
using cvec3 = glm::tvec3<char, glm::highp>;
using cvec4 = glm::tvec4<char, glm::highp>;

using ucvec2 = glm::tvec2<unsigned char, glm::highp>;
using ucvec3 = glm::tvec3<unsigned char, glm::highp>;
using ucvec4 = glm::tvec4<unsigned char, glm::highp>;
}  // namespace Saiga
#endif

//======= Output stream operator overloads =========

namespace glm
{
SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const vec4& v);
SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const dvec4& v);

SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const vec3& v);
SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const dvec3& v);

SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const vec2& v);


SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const mat3& v);
SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const mat4& v);
SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const dmat4& v);

SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const quat& v);

//======= Input stream operator overloads =========

SAIGA_CORE_API std::istream& operator>>(std::istream& is, vec2& v);
SAIGA_CORE_API std::istream& operator>>(std::istream& is, vec3& v);
SAIGA_CORE_API std::istream& operator>>(std::istream& is, vec4& v);
SAIGA_CORE_API std::istream& operator>>(std::istream& is, quat& v);

}  // namespace glm
//============== Helper functions =================


namespace Saiga
{
inline mat4 make_mat4(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13, float a20,
                      float a21, float a22, float a23, float a30, float a31, float a32, float a33)
{
    return mat4(a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33);
}

inline mat4 make_mat4(mat3 m)
{
    return mat4(m);
}

inline mat4 make_mat4(quat q)
{
    return mat4(q);
}

inline mat3 make_mat3(mat4 m)
{
    return mat3(m);
}


inline mat3 make_mat3(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13, float a20)
{
    return mat3(a00, a01, a02, a03, a10, a11, a12, a13, a20);
}

inline vec3 ele_mult(vec3 a, vec3 b)
{
    return a * b;
}


inline vec3 ele_div(vec3 a, vec3 b)
{
    return a / b;
}

inline vec2 ele_mult(vec2 a, vec2 b)
{
    return a * b;
}

inline vec4 ele_mult(vec4 a, vec4 b)
{
    return a * b;
}
inline const float* data(const mat4& m)
{
    return &m[0][0];
}
inline const float* data(const vec4& m)
{
    return &m[0];
}

inline vec4 make_vec4(const vec3& v, float a)
{
    return vec4(v, a);
}
inline vec3 make_vec3(const vec2& v, float a)
{
    return vec3(v, a);
}

inline vec4 make_vec4(float a)
{
    return vec4(a);
}

inline vec4 make_vec4(const vec2& v, const vec2& v2)
{
    return vec4(v, v2);
}

inline vec3 make_vec3(float a)
{
    return vec3(a);
}
inline vec3 make_vec3(vec4 a)
{
    return vec3(a);
}
inline vec2 make_vec2(float a)
{
    return vec2(a);
}
inline vec2 make_vec2(float a, float b)
{
    return vec2(a, b);
}

inline vec2 make_vec2(const ivec2& a)
{
    return vec2(a);
}
inline ivec2 make_ivec2(int a, int b)
{
    return ivec2(a, b);
}


inline vec2 make_vec2(vec3 a)
{
    return vec2(a);
}
inline vec4 quat_to_vec4(quat q)
{
    return vec4(q.x, q.y, q.z, q.w);
}
inline quat make_quat(float x, float y, float z, float w)
{
    // yes, the glm quat constructor is (w,x,y,z) even tho it is stored as (x,y,z,w)
    // GLM_FUNC_DECL GLM_CONSTEXPR qua(T w, T x, T y, T z);
    return quat(w, x, y, z);
}
inline quat make_quat(mat4 m)
{
    return quat(m);
}
inline mat4 identityMat4()
{
    return mat4(1);
}
inline mat3 identityMat3()
{
    return mat3(1);
}
inline ucvec4 make_ucvec4(const ucvec3& v, unsigned char a)
{
    return ucvec4(v, a);
}
inline vec4 make_vec4(float x, float y, float z, float w)
{
    return vec4(x, y, z, w);
}

inline const vec3& col(const mat3& m, int id)
{
    return m[id];
}

inline vec3& col(mat3& m, int id)
{
    return m[id];
}
inline vec4& col(mat4& m, int id)
{
    return m[id];
}
inline const vec4& col(const mat4& m, int id)
{
    return m[id];
}

inline float distance(vec3 a, vec3 b)
{
    return glm::distance(a, b);
}

inline mat4 zeroMat4()
{
    return mat4(0);
}
}  // namespace Saiga
namespace Saiga
{
SAIGA_CORE_API vec3 sampleCone(const vec3& dir, float angle);
// samples cone along the z axis
SAIGA_CORE_API vec3 sampleUnitCone(float angle);

SAIGA_CORE_API vec3 snapTo(vec3 v, float snapAngleInDegrees);


SAIGA_CORE_API inline mat4 createTRSmatrix(const vec4& t, const quat& r, const vec4& s)
{
    // Equivalent to:
    //    mat4 T = translate(mat4(1),vec3(t));
    //    mat4 R = mat4_cast(r);
    //    mat4 S = scale(mat4(1),vec3(s));
    //    return T * R * S;

    // Use optimized code here because T and S are sparse matrices and this
    // function is used pretty often.
    float qxx(r.x * r.x);
    float qyy(r.y * r.y);
    float qzz(r.z * r.z);
    float qxz(r.x * r.z);
    float qxy(r.x * r.y);
    float qyz(r.y * r.z);
    float qwx(r.w * r.x);
    float qwy(r.w * r.y);
    float qwz(r.w * r.z);

    mat4 Result(1 - 2 * (qyy + qzz), 2 * (qxy + qwz), 2 * (qxz - qwy), 0,

                2 * (qxy - qwz), 1 - 2 * (qxx + qzz), 2 * (qyz + qwx), 0,

                2 * (qxz + qwy), 2 * (qyz - qwx), 1 - 2 * (qxx + qyy), 0,

                t.x, t.y, t.z, 1);
    Result[0] *= s.x;
    Result[1] *= s.y;
    Result[2] *= s.z;
    return Result;
}

/**
 * Note:
 * The string conversion from and to matrices convert the glm matrix
 * to row major!!!
 */
SAIGA_CORE_API std::string to_string(const mat3& m);
SAIGA_CORE_API std::string to_string(const mat4& m);
SAIGA_CORE_API std::string to_string(const vec3& m);
SAIGA_CORE_API std::string to_string(const vec4& m);

SAIGA_CORE_API mat3 mat3FromString(const std::string& str);
SAIGA_CORE_API mat4 mat4FromString(const std::string& str);
SAIGA_CORE_API vec3 vec3FromString(const std::string& str);
SAIGA_CORE_API vec4 vec4FromString(const std::string& str);

}  // namespace Saiga
