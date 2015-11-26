#include "saiga/geometry/aabb.h"
#include <glm/gtc/epsilon.hpp>

aabb::aabb(void)
{
    min = glm::vec3(0,0,0);
    max = glm::vec3(0,0,0);
}

aabb::aabb(const glm::vec3 &p, const glm::vec3 &s) : min(p), max(s)
{
}

aabb::~aabb(void)
{
}

int aabb::maxDimension()
{
    vec3 d = max - min;

    float m = -1;
    int mi = -1;

    for(int i = 0 ; i < 3 ; ++i){
        if(d[i]>m){
            mi = i;
            m = d[i];
        }
    }
    return mi;
}

void aabb::transform(const mat4 &trafo){
    //only for scaling and translation correct !!!!
    min = vec3(trafo*vec4(min,1));
    max = vec3(trafo*vec4(max,1));
}

void aabb::makeNegative(){
#define INFINITE 100000000000000.0f
    min = vec3(INFINITE);
    max = vec3(-INFINITE);
}
#define MIN(X,Y) ((X<Y)?X:Y)
#define MAX(X,Y) ((X>Y)?X:Y)
#define MINV(V1,V2) vec3(MIN(V1.x,V2.x),MIN(V1.y,V2.y),MIN(V1.z,V2.z))
#define MAXV(V1,V2) vec3(MAX(V1.x,V2.x),MAX(V1.y,V2.y),MAX(V1.z,V2.z))
void aabb::growBox(const vec3 &v){

    min = MINV(min,v);
    max = MAXV(max,v);
}

void aabb::growBox(const aabb &v){
    min = MINV(min,v.min);
    max = MAXV(max,v.max);
}


void aabb::translate(const glm::vec3 &v)
{
    min += v;
    max += v;

}

void aabb::scale(const glm::vec3 &s){
    vec3 pos = getPosition();
    setPosition(vec3(0));
    min*=s;
    max*=s;
    setPosition(pos);
}

vec3 aabb::getPosition() const
{
    return 0.5f*(min+max);

}

void aabb::setPosition(const glm::vec3 &v)
{
    vec3 mid = 0.5f*(min+max);
    mid = v-mid;
    translate(mid);

}


void aabb::ensureValidity()
{
    float tmp;
    if(min.x > max.x){
        tmp = min.x;
        min.x = max.x;
        max.x = tmp;
    }

    if(min.y > max.y){
        tmp = min.y;
        min.y = max.y;
        max.y = tmp;
    }

    if(min.z > max.z){
        tmp = min.z;
        min.z = max.z;
        max.z = tmp;
    }

}

int aabb::touching(const aabb &other){

    for(int i = 0;i<3;i++){
        //glm::equalEpsilon(1,1,1);
        if(glm::epsilonEqual(max[i], other.min[i],0.001f) && intersect2(other,i)) return 0x8<<i;
        if(glm::epsilonEqual(min[i], other.max[i],0.001f) && intersect2(other,i)) return 0x1<<i;
    }
    return -1;

}

vec3 aabb::getHalfExtends()
{
    return 0.5f * (max-min);
}

bool aabb::intersect2(const aabb &other, int side){
    side = (side+1)%3;
    if(min[side] >= other.max[side] || max[side] <= other.min[side] ) return false;
    side = (side+1)%3;
    if(min[side] >= other.max[side] || max[side] <= other.min[side] ) return false;

    return true; //overlap
}

int aabb::intersect(const aabb &other){
    if(min.x >= other.max.x || max.x <= other.min.x ) return 0;
    if(min.y >= other.max.y || max.y <= other.min.y) return 0;
    if(min.z >= other.max.z || max.z <= other.min.z) return 0;

    if( other.min.x >= min.x && other.max.x <= max.x && //other inside this
            other.min.y >= min.y && other.max.y <= max.y &&
            other.min.z >= min.z && other.max.z <= max.z ) return 2; //contain
    return 1; //overlap
}

bool aabb::intersect2(const aabb &other){
    if(min.x >= other.max.x || max.x <= other.min.x ) return false;
    if(min.y >= other.max.y || max.y <= other.min.y) return false;
    if(min.z >= other.max.z || max.z <= other.min.z) return false;

    return true; //overlap
}

bool aabb::intersectTouching(const aabb &other){
    if(min.x > other.max.x || max.x < other.min.x ) return false;
    if(min.y > other.max.y || max.y < other.min.y) return false;
    if(min.z > other.max.z || max.z < other.min.z) return false;

    return true; //overlap

}

int aabb::intersectAabb(const aabb &other){
    if(min.x >= other.max.x || max.x <= other.min.x ) return 0;
    if(min.y >= other.max.y || max.y <= other.min.y) return 0;
    if(min.z >= other.max.z || max.z <= other.min.z) return 0;

    if( other.min.x >= min.x && other.max.x <= max.x && //other inside this
            other.min.y >= min.y && other.max.y <= max.y &&
            other.min.z >= min.z && other.max.z <= max.z ) return 2; //contain
    return 1; //overlap
}

bool aabb::intersectAabb2(const aabb &other){
    if(min.x >= other.max.x || max.x <= other.min.x ) return false;
    if(min.y >= other.max.y || max.y <= other.min.y) return false;
    if(min.z >= other.max.z || max.z <= other.min.z) return false;

    return true; //overlap
}



vec3 aabb::cornerPoint(int cornerIndex) const
{
    // assume(0 <= cornerIndex && cornerIndex <= 7);
    switch(cornerIndex)
    {
    default:
    case 0: return vec3(min.x, min.y, min.z);
    case 1: return vec3(min.x, min.y, max.z);
    case 2: return vec3(min.x, max.y, max.z);
    case 3: return vec3(min.x, max.y, min.z);
    case 4: return vec3(max.x, min.y, min.z);
    case 5: return vec3(max.x, min.y, max.z);
    case 6: return vec3(max.x, max.y, max.z);
    case 7: return vec3(max.x, max.y, min.z);
    }
}

bool aabb::contains(const glm::vec3 &p){
    if(min.x > p.x || max.x < p.x ) return false;
    if(min.y > p.y || max.y < p.y) return false;
    if(min.z > p.z || max.z < p.z) return false;

    return true; //overlap
}


std::ostream& operator<<(std::ostream& os, const aabb& bb)
{
    std::cout<<"aabb: ("<<bb.min.x<<","<<bb.min.y<<","<<bb.min.z<<")";
    std::cout<<" ("<<bb.max.x<<","<<bb.max.y<<","<<bb.max.z<<")";
    return os;
}
