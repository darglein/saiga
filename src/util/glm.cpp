#include "saiga/util/glm.h"

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

// ===========================================================================


SAIGA_GLOBAL std::istream& operator>>(std::istream& is, vec2& v){
    is >> v.x >> v.y;
    return is;
}

SAIGA_GLOBAL std::istream& operator>>(std::istream& is, vec3& v){
    is >> v.x >> v.y >> v.z;
    return is;
}

SAIGA_GLOBAL std::istream& operator>>(std::istream& is, vec4& v){
    is >> v.x >> v.y >> v.z >> v.w;
    return is;
}

SAIGA_GLOBAL std::istream& operator>>(std::istream& is, quat& v){
    is >> v.x >> v.y >> v.z >> v.w;
    return is;
}




// ===========================================================================

////TODO use glm::rotate
//quat glm::rotation(const vec3& v1, const vec3& v2){

//    //see http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another

//    float k_cos_theta = glm::dot(v1, v2);
//    float k = glm::sqrt(glm::length(v1) * glm::length(v2));

//    if (k_cos_theta / k == -1)
//    {
//      // 180 degree rotation around any orthogonal vector
//      vec3 other = (glm::abs(glm::dot(v1, vec3(1,0,0))) < 1.0) ? vec3(1,0,0) : vec3(0,1,0);
//      return glm::angleAxis(glm::pi<float>(),glm::normalize(glm::cross(v1, other)) );
//    }

//    return glm::normalize(quat(k_cos_theta + k, glm::cross(v1, v2)));
//}


vec3 sampleCone(const vec3 &dir, float angle){

    vec3 v = sampleUnitCone(angle);

    vec3 cdir = vec3(0,0,1);

//    if(dir==cdir){
//        return v;
//    }else if(dir==-cdir){
//        return -v;
//    }


    vec4 test = glm::rotation(cdir,dir)*vec4(v,0);

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


vec3 snapTo(vec3 v, float snapAngleInDegrees)
{
    vec3 snapAxis = vec3(1,0,0);
    float angle = glm::degrees(acos(glm::dot(v, snapAxis)));
    if (angle < snapAngleInDegrees / 2.0f) // Cannot do cross product
        return snapAxis * glm::length(v); // with angles 0 & 180
    if (angle > 180.0f - snapAngleInDegrees / 2.0f)
        return  -snapAxis* glm::length(v);
    float t = glm::round(angle / snapAngleInDegrees);

    float deltaAngle = (t * snapAngleInDegrees) - angle;

    vec3 axis = glm::cross(snapAxis, v);
    mat4 rot = glm::rotate(mat4(1),glm::radians(deltaAngle),axis);
    return vec3(rot * vec4(v,1));
}


void glmtest(){
    glm::linearRand(vec4(0),vec4(1));
    glm::diskRand(1.0f);
   glm::sphericalRand(1.0f);
    glm::ballRand(1.0f);
}

