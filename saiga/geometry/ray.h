#pragma once

#include <saiga/config.h>
#include "saiga/util/glm.h"
#include "saiga/geometry/aabb.h"
#include "saiga/geometry/sphere.h"
#include "saiga/geometry/triangle.h"
#include "saiga/geometry/plane.h"

namespace Saiga {

using glm::min;
using glm::max;


class SAIGA_GLOBAL Ray
{
public:
    vec3 direction;
    vec3 origin;
    vec3 dirfrac;
public:
	Ray(const vec3 &dir = vec3(0, 0, 0), const vec3 &ori = vec3(0, 0, 0));

    //[output] t: distance between ray origin and intersection
    bool intersectAabb(const AABB &bb, float &t) const;

     bool intersectSphere(const Sphere &s, float &t1, float &t2) const;

     bool intersectTriangle(const Triangle &s, float &t, bool &back) const;

     bool intersectPlane(const Plane& p, float &t) const;

    vec3 getAlphaPosition(float alpha) const { return origin+alpha*direction;}

    friend std::ostream& operator<<(std::ostream& os, const Ray& dt);
};

}
