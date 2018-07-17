/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/geometry/aabb.h"
#include <glm/gtc/epsilon.hpp>
#include "saiga/util/assert.h"
#include "internal/noGraphicsAPI.h"

namespace Saiga {

int AABB::maxDimension()
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

#define MIN(X,Y) ((X<Y)?X:Y)
#define MAX(X,Y) ((X>Y)?X:Y)
#define MINV(V1,V2) vec3(MIN(V1.x,V2.x),MIN(V1.y,V2.y),MIN(V1.z,V2.z))
#define MAXV(V1,V2) vec3(MAX(V1.x,V2.x),MAX(V1.y,V2.y),MAX(V1.z,V2.z))
void AABB::growBox(const vec3 &v){

    min = MINV(min,v);
    max = MAXV(max,v);
}

void AABB::growBox(const AABB &v){
    min = MINV(min,v.min);
    max = MAXV(max,v.max);
}



void AABB::ensureValidity()
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

int AABB::touching(const AABB &other){

    for(int i = 0;i<3;i++){
        //glm::equalEpsilon(1,1,1);
        if(glm::epsilonEqual(max[i], other.min[i],0.001f) && intersectBool(other,i)) return 0x8<<i;
        if(glm::epsilonEqual(min[i], other.max[i],0.001f) && intersectBool(other,i)) return 0x1<<i;
    }
    return -1;

}


bool AABB::intersectBool(const AABB &other, int side){
    side = (side+1)%3;
    if(min[side] >= other.max[side] || max[side] <= other.min[side] ) return false;
    side = (side+1)%3;
    if(min[side] >= other.max[side] || max[side] <= other.min[side] ) return false;

    return true; //overlap
}

int AABB::intersect(const AABB &other){
    if(min.x >= other.max.x || max.x <= other.min.x ) return 0;
    if(min.y >= other.max.y || max.y <= other.min.y) return 0;
    if(min.z >= other.max.z || max.z <= other.min.z) return 0;

    if( other.min.x >= min.x && other.max.x <= max.x && //other inside this
            other.min.y >= min.y && other.max.y <= max.y &&
            other.min.z >= min.z && other.max.z <= max.z ) return 2; //contain
    return 1; //overlap
}

bool AABB::intersectBool(const AABB &other){
    if(min.x >= other.max.x || max.x <= other.min.x ) return false;
    if(min.y >= other.max.y || max.y <= other.min.y) return false;
    if(min.z >= other.max.z || max.z <= other.min.z) return false;

    return true; //overlap
}

bool AABB::intersectTouching(const AABB &other){
    if(min.x > other.max.x || max.x < other.min.x ) return false;
    if(min.y > other.max.y || max.y < other.min.y) return false;
    if(min.z > other.max.z || max.z < other.min.z) return false;

    return true; //overlap

}


vec3 AABB::cornerPoint(int cornerIndex) const
{
     SAIGA_ASSERT(0 <= cornerIndex && cornerIndex <= 7);
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

bool AABB::contains(const vec3 &p){
    if(min.x > p.x || max.x < p.x ) return false;
    if(min.y > p.y || max.y < p.y) return false;
    if(min.z > p.z || max.z < p.z) return false;

    return true; //overlap
}

std::vector<Triangle> AABB::toTriangles()
{
    std::vector<Triangle> res = {
        //bottom
        Triangle( vec3(min.x,min.y,min.z) , vec3(max.x,min.y,min.z) , vec3(max.x,min.y,max.z) ),
        Triangle( vec3(min.x,min.y,min.z) , vec3(max.x,min.y,max.z) , vec3(min.x,min.y,max.z) ),

        //top
        Triangle( vec3(min.x,max.y,min.z) , vec3(max.x,max.y,min.z) , vec3(max.x,max.y,max.z) ),
        Triangle( vec3(min.x,max.y,min.z) , vec3(max.x,max.y,max.z) , vec3(min.x,max.y,max.z) ),


        //left
        Triangle( vec3(min.x,min.y,min.z) , vec3(min.x,min.y,max.z) , vec3(min.x,max.y,max.z) ),
        Triangle( vec3(min.x,min.y,min.z) , vec3(min.x,max.y,max.z) , vec3(min.x,max.y,min.z) ),

        //right
        Triangle( vec3(max.x,min.y,min.z) , vec3(max.x,min.y,max.z) , vec3(max.x,max.y,max.z) ),
        Triangle( vec3(max.x,min.y,min.z) , vec3(max.x,max.y,max.z) , vec3(max.x,max.y,min.z) ),


        //back
        Triangle( vec3(min.x,min.y,min.z) , vec3(min.x,max.y,min.z) , vec3(max.x,max.y,min.z) ),
        Triangle( vec3(min.x,min.y,min.z) , vec3(max.x,max.y,min.z) , vec3(max.x,min.y,min.z) ),


        //front
        Triangle( vec3(min.x,min.y,max.z) , vec3(min.x,max.y,max.z) , vec3(max.x,max.y,max.z) ),
        Triangle( vec3(min.x,min.y,max.z) , vec3(max.x,max.y,max.z) , vec3(max.x,min.y,max.z) )
    };

    return res;
}


std::ostream& operator<<(std::ostream& os, const AABB& bb)
{
     sampleCone(bb.min,5);
    std::cout<<"AABB: " << bb.min << " " << bb.max;
    return os;
}

}
