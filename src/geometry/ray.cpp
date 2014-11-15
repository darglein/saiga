#include "geometry/ray.h"

//using glm::min;
//using glm::max;
//source
//http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms

bool Ray::intersectAabb(const aabb &bb, float &t) const{
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



std::ostream& operator<<(std::ostream& os, const Ray& r)
{
    std::cout<<"Ray: ("<<r.origin.x<<","<<r.origin.y<<","<<r.origin.z<<")";
    std::cout<<" ("<<r.direction.x<<","<<r.direction.y<<","<<r.direction.z<<")";
    return os;
}
