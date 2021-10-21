/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "intersection.h"


namespace Saiga
{
namespace Intersection
{
inline bool PlanePlane(const Plane& p1, const Plane& p2, Ray& outRay)
{
    // similar to here
    // https://stackoverflow.com/questions/6408670/line-of-intersection-between-two-planes
    vec3 p3_normal = cross(p1.normal, p2.normal);
    auto det       = dot(p3_normal, p3_normal);
    if (det < epsilon<float>()) return false;
    auto outPoint = ((cross(p3_normal, p2.normal) * p1.d) + (cross(p1.normal, p3_normal) * p2.d)) / det;
    auto outDir   = p3_normal;
    outRay        = Ray(outDir, outPoint);
    return true;
}


inline bool RaySphere(const vec3& rayOrigin, const vec3& rayDir, const vec3& spherePos, float sphereRadius, float& t1,
                      float& t2)
{
    vec3 L  = rayOrigin - spherePos;
    float a = dot(rayDir, rayDir);
    float b = 2 * dot(rayDir, L);
    float c = dot(L, L) - sphereRadius * sphereRadius;
    float D = b * b + (-4.0f) * a * c;

    // rays misses sphere
    if (D < 0) return false;


    if (D == 0)
    {
        // ray touches sphere
        t1 = t2 = -0.5 * b / a;
    }
    else
    {
        // ray interescts sphere
        t1 = -0.5 * (b + sqrt(D)) / a;
        t2 = -0.5 * (b - sqrt(D)) / a;
    }

    if (t1 > t2)
    {
        auto tmp = t1;
        t1       = t2;
        t2       = tmp;
        //        std::swap(t1, t2);
    }
    return true;
}

inline bool RaySphere(const Ray& ray, const Sphere& sphere, float& t1, float& t2)
{
    return RaySphere(ray.origin, ray.direction, sphere.pos, sphere.r, t1, t2);
}


inline RayTriangleIntersection RayTriangle(const vec3& direction, const vec3& origin, const vec3& A, const vec3& B,
                                           const vec3& C, float epsilon)
{
    RayTriangleIntersection inter;


    vec3 e1, e2;  // Edge1, Edge2
    vec3 P, Q, T;
    float det, inv_det, u, v;
    float t;

    // Find vectors for two edges sharing V1
    e1 = B - A;
    e2 = C - A;

    // culling
    vec3 n         = cross(e1, e2);
    inter.backFace = dot(direction, n) > 0;

    // Begin calculating determinant - also used to calculate u parameter
    P = cross(direction, e2);
    // if determinant is near zero, ray lies in plane of triangle
    det = dot(e1, P);

    // NOT CULLING
    if (det > -epsilon && det < epsilon) return inter;
    inv_det = 1.f / det;

    // calculate distance from V1 to ray origin
    T = origin - A;

    // Calculate u parameter and test bound
    u = dot(T, P) * inv_det;
    // The intersection lies outside of the triangle
    if (u < 0.f || u > 1.f) return inter;

    // Prepare to test v parameter
    Q = cross(T, e1);

    // Calculate V parameter and test bound
    v = dot(direction, Q) * inv_det;

    // The intersection lies outside of the triangle
    if (v < 0.f || u + v > 1.f) return inter;

    t = dot(e2, Q) * inv_det;

    if (t > epsilon)
    {
        inter.valid = true;
        inter.t     = t;
        return inter;
    }

    return inter;
}


inline RayTriangleIntersection RayTriangle(const Ray& r, const Triangle& tri, float epsilon)
{
    const vec3& direction = r.direction;
    const vec3& origin    = r.origin;

    const vec3& A = tri.a;
    const vec3& B = tri.b;
    const vec3& C = tri.c;
    return RayTriangle(direction, origin, A, B, C, epsilon);
}

inline bool RayPlane(const Ray& r, const Plane& p, float& t)
{
    const vec3& direction = r.direction;
    const vec3& origin    = r.origin;

    const vec3& N = p.normal;
    const vec3& P = p.getPoint();


    const float EPSILON = 0.000001;

    float denom = dot(N, direction);

    // Check if ray is parallel to the plane
    if (abs(denom) > EPSILON)
    {
        t = dot(P - origin, N) / denom;
        if (t >= 0)
        {
            return true;
        }
    }
    return false;
}

// source
// http://gamedev.stackexchange.com/questions/18436/most-efficient-AABB-vs-ray-collision-algorithms
inline bool RayAABB(const vec3& origin, const vec3& direction, const vec3& boxmin, const vec3& boxmax, float& t)
{
    using std::max;
    using std::min;
    vec3 inv_dir = 1.0 / direction.array();

    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    vec3 t_min;
    t_min(0) = (boxmin[0] - origin[0]) * inv_dir[0];
    t_min(1) = (boxmin[1] - origin[1]) * inv_dir[1];
    t_min(2) = (boxmin[2] - origin[2]) * inv_dir[2];
    t_min    = (boxmin - origin).array() * inv_dir.array();

    vec3 t_max;
    t_max(0) = (boxmax[0] - origin[0]) * inv_dir[0];
    t_max(1) = (boxmax[1] - origin[1]) * inv_dir[1];
    t_max(2) = (boxmax[2] - origin[2]) * inv_dir[2];
    t_max    = (boxmax - origin).array() * inv_dir.array();

    // float tmin = std::max(max(min(t_min(0), t_max(0)), std::min(t_min(1), t_max(1))), std::min(t_min(2), t_max(2)));
    // float tmax = std::min(min(max(t_min(0), t_max(0)), std::max(t_min(1), t_max(1))), std::max(t_min(2), t_max(2)));

    float tmin = t_min.array().min(t_max.array()).maxCoeff();
    float tmax = t_min.array().max(t_max.array()).minCoeff();

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

inline bool RayAABB(const Ray& r, const AABB& bb, float& t)
{
    return RayAABB(r.origin, r.direction, bb.min, bb.max, t);
}


inline bool SphereSphere(const vec3& c1, float r1, const vec3& c2, float r2)
{
    return distance(c1, c2) < r1 + r2;
}

inline bool SphereSphere(const Sphere& s1, const Sphere& s2)
{
    return SphereSphere(s1.pos, s1.r, s2.pos, s2.r);
}

inline bool AABBAABB(const vec3& min1, const vec3& max1, const vec3& min2, const vec3& max2)
{
    if (min1[0] >= max2[0] || max1[0] <= min2[0]) return false;
    if (min1[1] >= max2[1] || max1[1] <= min2[1]) return false;
    if (min1[2] >= max2[2] || max1[2] <= min2[2]) return false;
    return true;
}

inline bool AABBAABB(const AABB& bb1, const AABB& bb2)
{
    return AABBAABB(bb1.min, bb1.max, bb2.min, bb2.max);
}



}  // namespace Intersection
}  // namespace Saiga
