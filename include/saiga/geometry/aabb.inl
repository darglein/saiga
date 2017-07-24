#include "saiga/geometry/aabb.h"


inline AABB::AABB()
{
}

inline AABB::AABB(const vec3 &min, const vec3 &max) : min(min), max(max)
{
}

inline AABB::~AABB()
{
}

inline void AABB::transform(const mat4 &trafo){
    //only for scaling and translation correct !!!!
    min = vec3(trafo*vec4(min,1));
    max = vec3(trafo*vec4(max,1));
}

inline void AABB::makeNegative(){
    const float INFINITE = 100000000000000.0f;
    min = vec3(INFINITE);
    max = vec3(-INFINITE);
}

inline void AABB::translate(const vec3 &v)
{
    min += v;
    max += v;
}

inline void AABB::scale(const vec3 &s){
    vec3 pos = getPosition();
    setPosition(vec3(0));
    min*=s;
    max*=s;
    setPosition(pos);
}

inline vec3 AABB::getPosition() const
{
    return 0.5f*(min+max);
}

inline void AABB::setPosition(const vec3 &v)
{
    vec3 mid = 0.5f*(min+max);
    mid = v-mid;
    translate(mid);
}

inline vec3 AABB::getHalfExtends()
{
    return 0.5f * (max-min);
}
