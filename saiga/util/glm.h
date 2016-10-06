#pragma once

#define GLM_FORCE_RADIANS

#include <saiga/config.h>

#include <string>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtx/quaternion.hpp>
//#include <glm/gtc/type_aligned.hpp>
#include <glm/gtx/type_aligned.hpp>

using std::ostream;

typedef glm::vec2 vec2;
typedef glm::vec3 vec3;

typedef glm::vec4 vec4;
typedef glm::mat4 mat4;
typedef glm::quat quat;
//typedef glm::aligned_vec4 vec4;
//typedef glm::aligned_mat4 mat4;
//typedef glm::aligned_quat quat;

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



SAIGA_GLOBAL inline mat4 createTRSmatrix(const vec3& translation, const quat& q, const vec3& scaling){
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
    Result[0] *= scaling[0];
    Result[1] *= scaling[1];
    Result[2] *= scaling[2];
    return Result;
}
