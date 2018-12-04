/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

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
using glm::linearRand;
using glm::max;
using glm::min;
using glm::mix;
using glm::radians;
using glm::sphericalRand;


#    define IDENTITY_QUATERNION quat(1, 0, 0, 0)

using cvec2 = glm::tvec2<char, glm::highp>;
using cvec3 = glm::tvec3<char, glm::highp>;
using cvec4 = glm::tvec4<char, glm::highp>;

using ucvec2 = glm::tvec2<unsigned char, glm::highp>;
using ucvec3 = glm::tvec3<unsigned char, glm::highp>;
using ucvec4 = glm::tvec4<unsigned char, glm::highp>;

#endif

//======= Output stream operator overloads =========

namespace glm
{
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const vec4& v);
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const dvec4& v);

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const vec3& v);
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const dvec3& v);

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const vec2& v);


SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const mat3& v);
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const mat4& v);
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const dmat4& v);

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const quat& v);

//======= Input stream operator overloads =========

SAIGA_GLOBAL std::istream& operator>>(std::istream& is, vec2& v);
SAIGA_GLOBAL std::istream& operator>>(std::istream& is, vec3& v);
SAIGA_GLOBAL std::istream& operator>>(std::istream& is, vec4& v);
SAIGA_GLOBAL std::istream& operator>>(std::istream& is, quat& v);

}  // namespace glm
//============== Helper functions =================


namespace Saiga
{
SAIGA_GLOBAL vec3 sampleCone(const vec3& dir, float angle);
// samples cone along the z axis
SAIGA_GLOBAL vec3 sampleUnitCone(float angle);

SAIGA_GLOBAL vec3 snapTo(vec3 v, float snapAngleInDegrees);



SAIGA_GLOBAL inline mat4 createTRSmatrix(const vec4& t, const quat& r, const vec4& s)
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
SAIGA_GLOBAL std::string to_string(const mat3& m);
SAIGA_GLOBAL std::string to_string(const mat4& m);
SAIGA_GLOBAL std::string to_string(const vec3& m);
SAIGA_GLOBAL std::string to_string(const vec4& m);

SAIGA_GLOBAL mat3 mat3FromString(const std::string& str);
SAIGA_GLOBAL mat4 mat4FromString(const std::string& str);
SAIGA_GLOBAL vec3 vec3FromString(const std::string& str);
SAIGA_GLOBAL vec4 vec4FromString(const std::string& str);

}  // namespace Saiga
