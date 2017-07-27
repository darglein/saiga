/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/geometry/intersection.h"

namespace Saiga {
namespace Intersection {

bool PlanePlane(const Plane &p1, const Plane &p2, Ray& outRay)
{
    //similar to here
    //https://stackoverflow.com/questions/6408670/line-of-intersection-between-two-planes
    vec3 p3_normal = cross(p1.normal,p2.normal);
    auto det = dot(p3_normal,p3_normal);
    if(det < glm::epsilon<float>())
        return false;
    auto outPoint = ((cross(p3_normal,p2.normal)  * p1.d) +
                ( cross(p1.normal,p3_normal) * p2.d)) / det;
    auto outDir = p3_normal;
    outRay = Ray(outDir,outPoint);
    return true;
}


bool RaySphere(const Ray& ray, const Sphere &sphere, float &t1, float &t2){
    vec3 L = ray.origin-sphere.pos;
    float a = glm::dot(ray.direction,ray.direction);
    float b = 2*glm::dot(ray.direction,L);
    float c = glm::dot(L,L) - sphere.r*sphere.r;
    float D = b*b + (-4.0f)*a*c;

    // If ray can not intersect then stop
    if (D < 0)
        return false;


    if(D==0){
        t1 = t2 = - 0.5 * b / a;
    }else{
        t1 = -0.5 * (b + sqrt(D)) / a ;
        t2 =  -0.5 * (b - sqrt(D)) / a;
    }
    if (t1 > t2) std::swap(t1, t2);


    return true;
}

}
}
