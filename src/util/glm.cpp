#include "util/glm.h"


std::ostream& operator<<(std::ostream& os, const vec4& v)
{
    os<<"("<<v.x<<","<<v.y<<","<<v.z<<","<<v.w<<")";
    return os;
}


std::ostream& operator<<(std::ostream& os, const glm::dvec4& v){
    os<<"("<<v.x<<","<<v.y<<","<<v.z<<","<<v.w<<")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const vec3& v)
{
    os<<"("<<v.x<<","<<v.y<<","<<v.z<<")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const glm::dvec3& v){
    os<<"("<<v.x<<","<<v.y<<","<<v.z<<")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const vec2& v)
{
    os<<"("<<v.x<<","<<v.y<<")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const mat4& v)
{
    for (int i = 0; i < 4; ++i){
        os << v[i] << "\n";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const glm::dmat4& v){
    for (int i = 0; i < 4; ++i){
        os << v[i] << "\n";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const quat& v){
    os<<"("<<v.x<<","<<v.y<<","<<v.z<<","<<v.w<<")";
    return os;
}

quat getRotation(const vec3& v1, const vec3& v2){
    vec3 rotAxis = glm::cross(v1,v2);
    float rotAngle = glm::acos(glm::dot(v1,v2));
    return glm::rotate(quat(),rotAngle,rotAxis);

}

vec3 sampleCone(const glm::vec3 &dir, float angle){
    vec3 v = sampleUnitCone(angle);

    vec4 test = getRotation(vec3(0,0,1),dir)*vec4(v,0);

    return vec3(test);

}

vec3 sampleUnitCone(float angle){
    float z = glm::linearRand(glm::cos(angle), float(1));
    float a = glm::linearRand(float(0), float(6.283185307179586476925286766559f));

    float r = glm::sqrt(float(1) - z * z);

    float x = r * cos(a);
    float y = r * sin(a);

    return vec3(x, y, z);
}
