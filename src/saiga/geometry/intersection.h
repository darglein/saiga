/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/util/math.h"

#include "aabb.h"
#include "plane.h"
#include "ray.h"
#include "sphere.h"
#include "triangle.h"

// TODO:
// move intersection methods from other classes to this file

namespace Saiga
{
namespace Intersection
{
/**
 * Intersection of 2 planes.
 * 2 general planes intersect in a line given by outDir and outPoint, unless they are parallel.
 * Returns false if the planes are parallel.
 */
SAIGA_GLOBAL bool PlanePlane(const Plane& p1, const Plane& p2, Ray& outRay);

/**
 * Intersection of a ray with a sphere.
 * There are either 2 intersections or 0, given by the return value.
 * t2 is always greater or equal to t1
 */
SAIGA_GLOBAL bool RaySphere(const vec3& rayOrigin, const vec3& rayDir, const vec3& spherePos, float sphereRadius,
                            float& t1, float& t2);
SAIGA_GLOBAL bool RaySphere(const Ray& ray, const Sphere& sphere, float& t1, float& t2);

/**
 * Intersection of a ray with a triangle.
 * There are either no interesection or exactly one at 't'.
 * 'back' is true, if the triangle was hit from behind (counter clockwise ordering)
 */
struct RayTriangleIntersection
{
    bool valid = false;
    float t    = std::numeric_limits<float>().infinity();  // position on ray
    bool backFace;
    int triangleIndex;  // usefull for raytracers

    bool operator<(const RayTriangleIntersection& other) { return t < other.t; }
    explicit operator bool() const { return valid; }
};
SAIGA_GLOBAL RayTriangleIntersection RayTriangle(const vec3& direction, const vec3& origin, const vec3& A,
                                                 const vec3& B, const vec3& C);
SAIGA_GLOBAL RayTriangleIntersection RayTriangle(const Ray& r, const Triangle& tri);



SAIGA_GLOBAL bool RayPlane(const Ray& r, const Plane& p, float& t);

SAIGA_GLOBAL bool RayAABB(const vec3& origin, const vec3& direction, const vec3& boxmin, const vec3& boxmax, float& t);
SAIGA_GLOBAL bool RayAABB(const Ray& r, const AABB& bb, float& t);


SAIGA_GLOBAL bool SphereSphere(const vec3& c1, float r1, const vec3& c2, float r2);
SAIGA_GLOBAL bool SphereSphere(const Sphere& s1, const Sphere& s2);

SAIGA_GLOBAL bool AABBAABB(const vec3& min1, const vec3& max1, const vec3& min2, const vec3& max2);
SAIGA_GLOBAL bool AABBAABB(const AABB& bb1, const AABB& bb2);

}  // namespace Intersection
}  // namespace Saiga
