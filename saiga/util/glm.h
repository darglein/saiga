#pragma once

#include <saiga/config.h>

#include <string>
#include <iostream>

//#define GLM_USE_SSE

#define GLM_FORCE_RADIANS

#if defined(GLM_USE_SSE)
//#define GLM_FORCE_ALIGNED
#define GLM_SIMD_ENABLE_XYZW_UNION
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtx/quaternion.hpp>
//#include <glm/gtx/type_aligned.hpp>

using std::ostream;



#if defined(GLM_USE_SSE)


//#include <glm/gtx/simd_quat.hpp>
//#include <glm/gtx/simd_vec4.hpp>
//#include <glm/gtx/simd_mat4.hpp>

#if GLM_VERSION != 98
#error GLM Version not supported!
#endif

//In version 98 this part is missing in glm/gtc/random.inl
namespace glm{
namespace detail{
template <template <class, precision> class vecType>
struct compute_linearRand<float, aligned_highp, vecType>
{
    GLM_FUNC_QUALIFIER static vecType<float, aligned_highp> call(vecType<float, aligned_highp> const & Min, vecType<float, aligned_highp> const & Max)
    {
        return vecType<float, aligned_highp>(compute_rand<uint32, aligned_highp, vecType>::call()) / static_cast<float>(std::numeric_limits<uint32>::max()) * (Max - Min) + Min;
    }
};
}
}

typedef glm::tvec2<float, glm::precision::aligned_highp> vec2;
typedef glm::tvec3<float, glm::precision::aligned_highp> vec3;
typedef glm::tvec4<float, glm::precision::aligned_highp> avec4;
typedef glm::tmat4x4<float, glm::precision::aligned_highp> amat4;
typedef glm::tquat<float, glm::precision::aligned_highp>  aquat;

//GLM_ALIGNED_TYPEDEF(avec2, vec2, 16);
//GLM_ALIGNED_TYPEDEF(avec3, vec3, 16);
GLM_ALIGNED_TYPEDEF(avec4, vec4, 16);
GLM_ALIGNED_TYPEDEF(amat4, mat4, 16);
GLM_ALIGNED_TYPEDEF(aquat, quat, 16);


//GLM_ALIGNED_TYPEDEF(glm::vec2, vec2, 16);
//GLM_ALIGNED_TYPEDEF(glm::vec3, vec3, 16);
//typedef glm::vec2 vec2;
//typedef glm::vec3 vec3;
//typedef glm::simdVec4 vec4;
//typedef glm::simdQuat quat;
//typedef glm::simdMat4 mat4;

//GLM_ALIGNED_TYPEDEF(glm::vec4, vec4, 16);
//GLM_ALIGNED_TYPEDEF(glm::mat4, mat4, 16);
//GLM_ALIGNED_TYPEDEF(glm::quat, quat, 16);

#else
typedef glm::vec2 vec2;
typedef glm::vec3 vec3;
typedef glm::vec4 vec4;
typedef glm::mat4 mat4;
typedef glm::quat quat;
#endif



using std::cout;
using std::endl;

//======= Output stream operator overloads =========

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const vec4& v);
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const glm::dvec4& v);

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const vec3& v);
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const glm::dvec3& v);

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const vec2& v);

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const mat4& v);
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const glm::dmat4& v);

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const quat& v);

//======= Input stream operator overloads =========

SAIGA_GLOBAL std::istream& operator>>(std::istream& is, vec2& v);
SAIGA_GLOBAL std::istream& operator>>(std::istream& is, vec3& v);
SAIGA_GLOBAL std::istream& operator>>(std::istream& is, vec4& v);
SAIGA_GLOBAL std::istream& operator>>(std::istream& is, quat& v);

//============== Helper functions =================



SAIGA_GLOBAL vec3 sampleCone(const vec3& dir, float angle);
//samples cone along the z axis
SAIGA_GLOBAL vec3 sampleUnitCone(float angle);

SAIGA_GLOBAL vec3 snapTo(vec3 v, float snapAngleInDegrees);



SAIGA_GLOBAL inline mat4 createTRSmatrix(const vec4& translation, const quat& q, const vec4& scaling){
    //equivalent to:
    //    mat4 matrix = mat4_cast(rotation);
    //    matrix[0] *= scaling[0];
    //    matrix[1] *= scaling[1];
    //    matrix[2] *= scaling[2];
    //    matrix[3] = vec4(translation,1);
    float qxx(q.x * q.x);
    float qyy(q.y * q.y);
    float qzz(q.z * q.z);
    float qxz(q.x * q.z);
    float qxy(q.x * q.y);
    float qyz(q.y * q.z);
    float qwx(q.w * q.x);
    float qwy(q.w * q.y);
    float qwz(q.w * q.z);

    mat4 Result(
                1 - 2 * (qyy +  qzz),
                2 * (qxy + qwz),
                2 * (qxz - qwy),
                0,

                2 * (qxy - qwz),
                1 - 2 * (qxx +  qzz),
                2 * (qyz + qwx),
                0,

                2 * (qxz + qwy),
                2 * (qyz - qwx),
                1 - 2 * (qxx +  qyy),
                0,

                translation.x,
                translation.y,
                translation.z,
                1
                );
    Result[0] *= scaling.x;
    Result[1] *= scaling.y;
    Result[2] *= scaling.z;
    return Result;
}
