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
#include "glm/gtx/quaternion.hpp"

using std::ostream;

using glm::mat4;
using glm::vec3;
using glm::vec4;
using glm::vec2;
using glm::quat;
using std::cout;
using std::endl;
#define degreesToRadians(x) x*(3.141592f/180.0f)

//======= Output stream operator overloads =========

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const glm::vec4& v);
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const glm::dvec4& v);

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const glm::vec3& v);
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const glm::dvec3& v);

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const glm::vec2& v);

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const glm::mat4& v);
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const glm::dmat4& v);

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const glm::quat& v);

//======= Input stream operator overloads =========

SAIGA_GLOBAL std::istream& operator>>(std::istream& is, glm::vec2& v);
SAIGA_GLOBAL std::istream& operator>>(std::istream& is, glm::vec3& v);
SAIGA_GLOBAL std::istream& operator>>(std::istream& is, glm::vec4& v);
SAIGA_GLOBAL std::istream& operator>>(std::istream& is, glm::quat& v);

//============== Helper functions =================

//returns quaternion that rotates v1 to v2
SAIGA_GLOBAL inline glm::quat getRotation(const glm::vec3& v1, const glm::vec3& v2){return glm::rotation(v1,v2);}


SAIGA_GLOBAL glm::vec3 sampleCone(const glm::vec3& dir, float angle);
//samples cone along the z axis
SAIGA_GLOBAL glm::vec3 sampleUnitCone(float angle);

SAIGA_GLOBAL glm::vec3 snapTo(glm::vec3 v, float snapAngleInDegrees);



SAIGA_GLOBAL inline glm::mat4 createTRSmatrix(const vec3& translation, const quat& rotation, const vec3& scaling){
    glm::mat4 matrix = glm::mat4_cast(rotation);
    matrix[0] *= scaling[0];
    matrix[1] *= scaling[1];
    matrix[2] *= scaling[2];
    matrix[3] = vec4(translation,1);
    return matrix;
}
