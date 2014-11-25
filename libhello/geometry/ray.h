#pragma once
#include "libhello/util/glm.h"
#include "libhello/geometry/aabb.h"
#include "libhello/geometry/sphere.h"
#include "libhello/geometry/triangle.h"

using glm::min;
using glm::max;


class Ray
{
public:
    vec3 direction;
    vec3 origin;
    vec3 dirfrac;
public:
    Ray(const vec3 &dir=vec3(0,0,0),const vec3 &ori=vec3(0,0,0)):direction(dir),origin(ori){
        dirfrac.x = 1.0f / direction.x;
        dirfrac.y = 1.0f / direction.y;
        dirfrac.z = 1.0f / direction.z;
    }

    //[output] t: distance between ray origin and intersection
    bool intersectAabb(const aabb &bb, float &t) const;

     bool intersectSphere(const Sphere &s, float &t1, float &t2) const;

     bool intersectTriangle(const Triangle &s, float &t, bool &back) const;

    vec3 getAlphaPosition(float alpha) const { return origin+alpha*direction;}

    friend std::ostream& operator<<(std::ostream& os, const Ray& dt);
};

