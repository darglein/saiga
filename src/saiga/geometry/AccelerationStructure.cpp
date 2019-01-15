/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "AccelerationStructure.h"

namespace Saiga
{
namespace AccelerationStructure
{
BruteForce::BruteForce(const std::vector<Saiga::Triangle>& triangles) : triangles(triangles) {}

RayTriangleIntersection BruteForce::getClosest(const Ray& ray)
{
    Intersection::RayTriangleIntersection result;
    for (size_t i = 0; i < triangles.size(); ++i)
    {
        auto& tri  = triangles[i];
        auto inter = Intersection::RayTriangle(ray, tri);
        if (inter && inter < result)
        {
            inter.triangleIndex = i;
            result              = inter;
        }
    }
    return result;
}

std::vector<RayTriangleIntersection> BruteForce::getAll(const Ray& ray)
{
    std::vector<RayTriangleIntersection> result;
    for (size_t i = 0; i < triangles.size(); ++i)
    {
        auto& tri  = triangles[i];
        auto inter = Intersection::RayTriangle(ray, tri);
        if (inter)
        {
            inter.triangleIndex = i;
            result.push_back(inter);
        }
    }
    return result;
}


BVH::BVH(const std::vector<Saiga::Triangle>& triangles) : triangles(triangles)
{
    static_assert(sizeof(BVHNode) == 8 * sizeof(float), "Node size broken.");
}

RayTriangleIntersection BVH::getClosest(const Ray& ray)
{
    Intersection::RayTriangleIntersection result;
    if (!nodes.empty()) getClosest(0, ray, result);
    return result;
}

std::vector<Intersection::RayTriangleIntersection> BVH::getAll(const Ray& ray)
{
    std::vector<RayTriangleIntersection> result;
    return result;
}

void BVH::getClosest(int node, const Ray& ray, Intersection::RayTriangleIntersection& result)
{
    BVHNode& n = nodes[node];

    float aabbT;

    // The ray missed the box
    if (!Intersection::RayAABB(ray, n.box, aabbT)) return;

    // The node is further than the closest hit
    if (aabbT > result.t) return;

    if (n._inner)
    {
        getClosest(n._left, ray, result);
        getClosest(n._right, ray, result);
    }
    else
    {
        // Leaf node -> intersect with triangles
        for (uint32_t i = n._left; i < n._right; ++i)
        {
            auto inter = Intersection::RayTriangle(ray, triangles[i]);
            if (inter && inter < result)
            {
                inter.triangleIndex = i;
                result              = inter;
            }
        }
    }
}


}  // namespace AccelerationStructure
}  // namespace Saiga
