/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

namespace Saiga
{
/**
 * Simple class for a 3D triangle consiting of the 3 corner points a, b, and c.
 * @brief The Triangle class
 */

class SAIGA_CORE_API Triangle
{
   public:
    vec3 a, b, c;

   public:
    Triangle() {}
    Triangle(const vec3& a, const vec3& b, const vec3& c) : a(a), b(b), c(c) {}


    // Scale this triangle uniformly by the given factor.
    //  - Translate by -center
    //  - Multiply by f
    //  - Translate by center
    // Something like t.ScaleUniform(1.0005) can be used to achieve a slight overlap during raytracing.
    void ScaleUniform(float f);
    /**
     * Mean center: (a+b+c) / 3
     * @brief center
     * @return
     */
    vec3 center() const;

    /**
     * Computes and returns the minial angle of the corners.
     * Usefull to check for degenerate triangles.
     * @brief minimalAngle
     * @return
     */
    float minimalAngle() const;
    float cosMinimalAngle() const;

    /**
     * Computes the inner angle at a triangle corner.
     * i = 0 at corner a
     * i = 1 at corner b
     * i = 2 at corner c
     * @brief angleAtCorner
     * @param i
     * @return
     */
    float angleAtCorner(int i) const;
    float cosAngleAtCorner(int i) const;

    /**
     * Check if this triangle is broken.
     */
    bool isDegenerate() const;

    float Area() const;

    vec3 RandomPointOnSurface() const;

    /**
     * Computes the normal with a cross product.
     * The positive side is with counter-clock-wise ordering.
     */
    vec3 normal() const;

    // Returns the barycentric coordinates of point x projected to the triangle plane
    vec3 BarycentricCoordinates(const vec3& x) const;

    // Distance of a point to the triangle
    float Distance(const vec3& x) const;

    friend SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const Triangle& dt);
};

}  // namespace Saiga
