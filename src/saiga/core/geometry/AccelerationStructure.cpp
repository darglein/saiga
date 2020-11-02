/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "AccelerationStructure.h"

#include "algorithm"

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


BVH::BVH(const std::vector<Saiga::Triangle>& triangles)
{
    static_assert(sizeof(BVHNode) == 8 * sizeof(float), "Node size broken.");
    for (size_t i = 0; i < triangles.size(); ++i)
    {
        this->triangles.push_back({triangles[i], i});
    }
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
    if (!nodes.empty()) getAll(0, ray, result);
    return result;
}

AABB BVH::computeBox(int start, int end)
{
    AABB box;
    box.makeNegative();
    for (int i = start; i < end; ++i)
    {
        auto& t = triangles[i].first;
        box.growBox(t.a);
        box.growBox(t.b);
        box.growBox(t.c);
    }
    return box;
}

void BVH::sortByAxis(int start, int end, int axis)
{
    std::sort(triangles.begin() + start, triangles.begin() + end, SortTriangleByAxis(axis));
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
            auto inter = Intersection::RayTriangle(ray, triangles[i].first);
            if (inter && inter < result)
            {
                inter.triangleIndex = triangles[i].second;
                result              = inter;
            }
        }
    }
}

void BVH::getAll(int node, const Ray& ray, std::vector<Intersection::RayTriangleIntersection>& result)
{
    BVHNode& n = nodes[node];

    float aabbT;

    // The ray missed the box
    if (!Intersection::RayAABB(ray, n.box, aabbT)) return;

    if (n._inner)
    {
        getAll(n._left, ray, result);
        getAll(n._right, ray, result);
    }
    else
    {
        // Leaf node -> intersect with triangles
        for (uint32_t i = n._left; i < n._right; ++i)
        {
            auto inter = Intersection::RayTriangle(ray, triangles[i].first);
            if (inter)
            {
                inter.triangleIndex = triangles[i].second;
                result.push_back(inter);
            }
        }
    }
}

void ObjectMedianBVH::construct()
{
    nodes.reserve(triangles.size());
    construct(0, triangles.size());
}

int ObjectMedianBVH::construct(int start, int end)
{
    int nodeid = nodes.size();
    nodes.push_back({});
    auto& node = nodes.back();

    node.box = computeBox(start, end);

    if (end - start <= leafTriangles)
    {
        // leaf node
        node._inner = 0;
        node._left  = start;
        node._right = end;
    }
    else
    {
        node._inner = 1;
        int axis    = node.box.maxDimension();
        sortByAxis(start, end, axis);

        int mid = (start + end) / 2;

        int l = construct(start, mid);
        int r = construct(mid, end);

        // reload node, because the reference from above might be broken
        auto& node2  = nodes[nodeid];
        node2._left  = l;
        node2._right = r;
    }

    return nodeid;
}


}  // namespace AccelerationStructure
}  // namespace Saiga
