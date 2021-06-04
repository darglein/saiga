/**
 * Copyright (c) 2021 Darius Rückert
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
 *
 * This class can be used in CUDA, however not all member functions are available there.
 * @brief The Triangle class
 */
class SAIGA_CORE_API Triangle
{
   public:
    vec3 a, b, c;

   public:
    HD Triangle() {}
    HD Triangle(const vec3& a, const vec3& b, const vec3& c) : a(a), b(b), c(c) {}

    /**
     * Mean center: (a+b+c) / 3
     * @brief center
     * @return
     */
    HD vec3 center() const { return (a + b + c) * float(1.0f / 3.0f); }

    HD float Area() const { return 0.5f * cross(b - a, c - a).norm(); }


    /**
     * Computes the normal with a cross product.
     * The positive side is with counter-clock-wise ordering.
     */
    HD vec3 normal() const { return normalize(cross(b - a, c - a)); }


    // ================== Defined in triangle.cpp ==================
    // These function are currently not usable in cuda.
    // =============================================================

    vec3 RandomPointOnSurface() const;
    vec3 RandomBarycentric() const;


    // Scale this triangle uniformly by the given factor.
    //  - Translate by -center
    //  - Multiply by f
    //  - Translate by center
    // Something like t.ScaleUniform(1.0005) can be used to achieve a slight overlap during raytracing.
    void ScaleUniform(float f);

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


    // Returns the barycentric coordinates of point x projected to the triangle plane
    vec3 BarycentricCoordinates(const vec3& x) const;

    vec3 InterpolateBarycentric(const vec3& bary) const;

    // Distance of a point to the triangle
    float Distance(const vec3& x) const;

    friend SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const Triangle& dt);
};

}  // namespace Saiga
