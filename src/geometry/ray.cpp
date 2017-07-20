/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/geometry/ray.h"

namespace Saiga {

Ray::Ray(const vec3 &dir , const vec3 &ori) :direction(dir), origin(ori){
	dirfrac.x = 1.0f / direction.x;
	dirfrac.y = 1.0f / direction.y;
	dirfrac.z = 1.0f / direction.z;
}


//using glm::min;
//using glm::max;
//source
//http://gamedev.stackexchange.com/questions/18436/most-efficient-AABB-vs-ray-collision-algorithms

bool Ray::intersectAabb(const AABB &bb, float &t) const{
    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    float t1 = (bb.min.x - origin.x)*dirfrac.x;
    float t2 = (bb.max.x - origin.x)*dirfrac.x;
    float t3 = (bb.min.y - origin.y)*dirfrac.y;
    float t4 = (bb.max.y - origin.y)*dirfrac.y;
    float t5 = (bb.min.z - origin.z)*dirfrac.z;
    float t6 = (bb.max.z - origin.z)*dirfrac.z;

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us
    if (tmax < 0)
    {
        t = tmax;
        return false;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax)
    {
        t = tmax;
        return false;
    }

    t = tmin;
    return true;
}

bool Ray::intersectSphere(const Sphere &s, float &t1, float &t2) const{
    vec3 L = origin-s.pos;
    float a = glm::dot(direction,direction);
    float b = 2*glm::dot(direction,( L));
    float c = glm::dot(L,L) - s.r*s.r;
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

bool Ray::intersectTriangle(const Triangle &tri, float &out, bool &back) const{
    const float EPSILON = 0.000001f;
    vec3 e1, e2;  //Edge1, Edge2
    vec3 P, Q, T;
    float det, inv_det, u, v;
    float t;

    //Find vectors for two edges sharing V1
    e1 = tri.b - tri.a;
    e2 = tri.c-tri.a;

    //culling
    vec3 n = glm::cross(e1,e2);
    back = glm::dot(direction,n)>0;

    //Begin calculating determinant - also used to calculate u parameter
    P = glm::cross( direction, e2);
    //if determinant is near zero, ray lies in plane of triangle
    det = glm::dot(e1, P);

    //NOT CULLING
    if(det > -EPSILON && det < EPSILON) return 0;
    inv_det = 1.f / det;

    //calculate distance from V1 to ray origin
    T=origin-tri.a;

    //Calculate u parameter and test bound
    u = glm::dot(T, P) * inv_det;
    //The intersection lies outside of the triangle
    if(u < 0.f || u > 1.f) return 0;

    //Prepare to test v parameter
    Q = glm::cross( T, e1);

    //Calculate V parameter and test bound
    v = glm::dot(direction, Q) * inv_det;
    //The intersection lies outside of the triangle
    if(v < 0.f || u + v  > 1.f) return 0;

    t = glm::dot(e2, Q) * inv_det;

    if(t > EPSILON) { //ray intersection
        out = t;
        return 1;
    }

    // No hit, no win
    return 0;
}

bool Ray::intersectPlane(const Plane &p, float &t) const
{
    const float EPSILON = 0.000001;

    float denom = glm::dot(p.normal, direction);

    if (glm::abs(denom) > EPSILON)
    {
        t = glm::dot(p.point - origin, p.normal) / denom;
        if (t >= 0){
            return true;
        }
    }
    return false;
}

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const Ray& r)
{
    std::cout<<"Ray: " << r.origin << " " << r.direction;
    return os;
}

}
