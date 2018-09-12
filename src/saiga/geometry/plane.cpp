/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/geometry/plane.h"
#include "internal/noGraphicsAPI.h"
namespace Saiga {


Plane::Plane(const vec3 &point,const vec3 &normal){
    set(point,normal);
}

Plane::Plane(const vec3 &p1, const vec3 &p2, const vec3 &p3){
    set(p1,p2,p3);
}

void Plane::set(const vec3 &point,const vec3 &normal)
{

    this->normal = glm::normalize(normal);
    d = glm::dot(point,this->normal);
}

void Plane::set(const vec3 &p1, const vec3 &p2, const vec3 &p3)
{
    normal = glm::cross(p2-p1,p3-p1);
    normal = glm::normalize(normal);
    d = glm::dot(p1,this->normal);
}



vec3 Plane::closestPointOnPlane(const vec3 &p) const
{
    float dis = distance(p);
    return p - dis * normal;
}

vec3 Plane::getPoint() const
{
    return normal * d;
}


std::ostream& operator<<(std::ostream& os, const Plane& pl){

    os << "x * " << pl.normal << " - " << pl.d << " = 0";
    return os;
}

}
