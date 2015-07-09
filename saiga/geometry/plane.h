#pragma once

#include <saiga/config.h>
#include "saiga/util/glm.h"

class SAIGA_GLOBAL Plane
{
public:
    vec3 point; //a random point on the plane
    vec3 normal;
    float d; //distance from plane to origin
    Plane();
    Plane(const vec3 &point,const vec3 &normal);
    Plane(const vec3 &p1, const vec3 &p2, const vec3 &p3); //construct plane from 3 points

    void set(const vec3 &point,const vec3 &normal);
    void set(const vec3 &p1, const vec3 &p2, const vec3 &p3);

    float distance(const vec3 &p) const;

    void draw();



   friend std::ostream& operator<<(std::ostream& os, const Plane& ca);
};


