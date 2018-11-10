/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/util/math.h"

namespace Saiga {

/**
 * Simple class for a 3D triangle consiting of the 3 corner points a, b, and c.
 * @brief The Triangle class
 */

class SAIGA_GLOBAL Triangle
{
public:
    vec3 a,b,c;
public:
    Triangle(){}
    Triangle(const vec3 &a, const vec3 &b, const vec3 &c):a(a),b(b),c(c){ }

    /**
     * Mean center: (a+b+c) / 3
     * @brief center
     * @return
     */
    vec3 center();

    /**
     * Computes and returns the minial angle of the corners.
     * Usefull to check for degenerate triangles.
     * @brief minimalAngle
     * @return
     */
    float minimalAngle();
    float cosMinimalAngle();

    /**
     * Computes the inner angle at a triangle corner.
     * i = 0 at corner a
     * i = 1 at corner b
     * i = 2 at corner c
     * @brief angleAtCorner
     * @param i
     * @return
     */
    float angleAtCorner(int i);
    float cosAngleAtCorner(int i);

    /**
     * Check if this triangle is broken.
     */
    bool isDegenerate();


    /**
     * Computes the normal with a cross product.
     * The positive side is with counter-clock-wise ordering.
     */
    vec3 normal();

    friend SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const Triangle& dt);
};

}
