#pragma once

#include <saiga/config.h>
#include "saiga/util/glm.h"

using glm::min;
using glm::max;


class SAIGA_GLOBAL Triangle
{
public:
    vec3 a,b,c;
public:
    Triangle(){}
    Triangle(const vec3 &a, const vec3 &b, const vec3 &c):a(a),b(b),c(c){ }

    void stretch();

    vec3 center();


    friend std::ostream& operator<<(std::ostream& os, const Triangle& dt);
};

