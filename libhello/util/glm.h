#ifndef GLM_H
#define GLM_H

#include <string>
#include <iostream>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
//#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <glm/gtc/quaternion.hpp>

using std::ostream;

using glm::mat4;
using glm::vec3;
using glm::vec4;
using glm::vec2;
using glm::quat;
#define degreesToRadians(x) x*(3.141592f/180.0f)



std::ostream& operator<<(std::ostream& os, const vec4& v);
std::ostream& operator<<(std::ostream& os, const glm::dvec4& v);

std::ostream& operator<<(std::ostream& os, const vec3& v);
std::ostream& operator<<(std::ostream& os, const glm::dvec3& v);

std::ostream& operator<<(std::ostream& os, const vec2& v);

std::ostream& operator<<(std::ostream& os, const mat4& v);
std::ostream& operator<<(std::ostream& os, const glm::dmat4& v);

std::ostream& operator<<(std::ostream& os, const quat& v);

#endif // GLM_H
