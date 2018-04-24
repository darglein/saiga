/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/util/glm.h"

namespace Saiga {

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

    /**
     *  Uses first point as plane point and computes normal via cross product.
     *  Similar to triangles the points should be ordered counter clock wise to give a positive normal.
     */
    void set(const vec3 &p1, const vec3 &p2, const vec3 &p3);

    float distance(const vec3 &p) const;
    vec3 closestPointOnPlane(const vec3 &p) const;




    friend std::ostream& operator<<(std::ostream& os, const Plane& ca);
};

}
