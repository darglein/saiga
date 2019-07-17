/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"

#ifdef SAIGA_USE_GLM
#include "glm.h"

#include "saiga/core/util/assert.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

namespace glm
{
std::ostream& operator<<(std::ostream& os, const vec4& v)
{
    os << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")";
    return os;
}


std::ostream& operator<<(std::ostream& os, const dvec4& v)
{
    os << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const vec3& v)
{
    os << "(" << v.x << "," << v.y << "," << v.z << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const dvec3& v)
{
    os << "(" << v.x << "," << v.y << "," << v.z << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const vec2& v)
{
    os << "(" << v.x << "," << v.y << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const mat3& v)
{
    auto vt = transpose(v);
    for (int i = 0; i < 3; ++i)
    {
        os << vt[i] << std::endl;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const mat4& v)
{
    auto vt = transpose(v);
    for (int i = 0; i < 4; ++i)
    {
        os << vt[i] << std::endl;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const dmat4& v)
{
    for (int i = 0; i < 4; ++i)
    {
        os << v[i] << "\n";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const quat& v)
{
    os << "(" << v.w << "," << v.x << "," << v.y << "," << v.z << ")";
    return os;
}

// ===========================================================================


SAIGA_CORE_API std::istream& operator>>(std::istream& is, vec2& v)
{
    is >> v.x >> v.y;
    return is;
}

SAIGA_CORE_API std::istream& operator>>(std::istream& is, vec3& v)
{
    is >> v.x >> v.y >> v.z;
    return is;
}

SAIGA_CORE_API std::istream& operator>>(std::istream& is, vec4& v)
{
    is >> v.x >> v.y >> v.z >> v.w;
    return is;
}

SAIGA_CORE_API std::istream& operator>>(std::istream& is, quat& v)
{
    is >> v.x >> v.y >> v.z >> v.w;
    return is;
}


}  // namespace glm

namespace Saiga
{
// ===========================================================================

////TODO use rotate
// quat rotation(const vec3& v1, const vec3& v2){

//    //see
//    http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another

//    float k_cos_theta = dot(v1, v2);
//    float k = sqrt(length(v1) * length(v2));

//    if (k_cos_theta / k == -1)
//    {
//      // 180 degree rotation around any orthogonal vector
//      vec3 other = (abs(dot(v1, vec3(1,0,0))) < 1.0) ? vec3(1,0,0) : vec3(0,1,0);
//      return angleAxis(pi<float>(),normalize(cross(v1, other)) );
//    }

//    return normalize(quat(k_cos_theta + k, cross(v1, v2)));
//}


vec3 sampleCone(const vec3& dir, float angle)
{
    vec3 v = sampleUnitCone(angle);

    vec3 cdir = vec3(0, 0, 1);

    //    if(dir==cdir){
    //        return v;
    //    }else if(dir==-cdir){
    //        return -v;
    //    }


    vec4 test = rotation(cdir, dir) * vec4(v, 0);

    return vec3(test);
}

vec3 sampleUnitCone(float angle)
{
    float z = linearRand((float)cos(angle), float(1));
    float a = linearRand(float(0), float(6.283185307179586476925286766559f));

    float r = sqrt(float(1) - z * z);

    float x = r * cos(a);
    float y = r * sin(a);

    return vec3(x, y, z);
}


vec3 snapTo(vec3 v, float snapAngleInDegrees)
{
    vec3 snapAxis = vec3(1, 0, 0);
    float angle   = degrees(acos(dot(v, snapAxis)));
    if (angle < snapAngleInDegrees / 2.0f)  // Cannot do cross product
        return snapAxis * length(v);        // with angles 0 & 180
    if (angle > 180.0f - snapAngleInDegrees / 2.0f) return -snapAxis * length(v);
    float t = round(angle / snapAngleInDegrees);

    float deltaAngle = (t * snapAngleInDegrees) - angle;

    vec3 axis = cross(snapAxis, v);
    mat4 rot  = rotate(mat4(1), radians(deltaAngle), axis);
    return vec3(rot * vec4(v, 1));
}



template <typename M>
inline std::string helpConvert(const M& m, int size)
{
    auto a = make_ArrayView((float*)&m, size);
    return array_to_string(a);
}

template <typename M>
inline M helpConvertBack(const std::string& str, unsigned int size)
{
    auto array = string_to_array<float>(str);
    SAIGA_ASSERT(array.size() == size);
    M m(1);
    std::copy(array.begin(), array.end(), (float*)&m);
    return m;
}
std::string to_string(const mat3& m)
{
    return helpConvert(transpose(m), 16);
}
std::string to_string(const mat4& m)
{
    return helpConvert(transpose(m), 16);
}
std::string to_string(const vec3& m)
{
    return helpConvert(m, 3);
}
std::string to_string(const vec4& m)
{
    return helpConvert(m, 4);
}

mat3 mat3FromString(const std::string& str)
{
    return transpose(helpConvertBack<mat3>(str, 9));
}
mat4 mat4FromString(const std::string& str)
{
    return transpose(helpConvertBack<mat4>(str, 16));
}
vec3 vec3FromString(const std::string& str)
{
    return helpConvertBack<vec3>(str, 3);
}
vec4 vec4FromString(const std::string& str)
{
    return helpConvertBack<vec4>(str, 4);
}

}  // namespace Saiga
#endif
