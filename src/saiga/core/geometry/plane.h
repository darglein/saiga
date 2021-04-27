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
 * Implicit representation of a plane.
 * This class is save for use in CUDA kernels.
 *
 * x * n - d = 0
 *
 * With:
 * x: a point
 * n: the plane normal
 * d: the distance to the origin (projected to n)
 */
class SAIGA_TEMPLATE Plane
{
   public:
    vec3 normal;
    float d;

    HD Plane();
    HD Plane(const vec3& point, const vec3& normal);

    /**
     *  Uses first point as plane point and computes normal via cross product.
     *  Similar to triangles the points should be ordered counter clock wise to give a positive normal.
     */
    HD Plane(const vec3& p1, const vec3& p2, const vec3& p3);

    /**
     * Returns the plane with inverted normal and offset.
     */
    HD Plane invert() const;

    /**
     * (Signed) Distance from the point 'p' to the plane.
     */
    HD float distance(const vec3& p) const;

    /**
     * The overlapping distance between a sphere and this plane.
     * Negative if the sphere does NOT intersect the plane.
     */
    HD float sphereOverlap(const vec3& c, float r) const;

    /**
     * The intersecting circle of a sphere on this plane.
     * Sphere center is projected on the plane and the radius is calculated.
     */
    std::pair<vec3, float> intersectingCircle(const vec3& c, float r) const;

    /**
     * Returns the point on the plane which is closest to the given point p.
     */
    HD vec3 closestPointOnPlane(const vec3& p) const;

    /**
     * Returns the point on the plane which is closest to the origin.
     */
    HD vec3 getPoint() const;


    friend std::ostream& operator<<(std::ostream& os, const Plane& plane);
};

}  // namespace Saiga

#include "saiga/core/geometry/plane.inl"
