#include "util/glm.h"


std::ostream& operator<<(std::ostream& os, const vec4& v)
{
    os<<"("<<v.x<<","<<v.y<<","<<v.z<<","<<v.w<<")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const vec3& v)
{
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
