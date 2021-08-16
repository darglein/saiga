/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

#include "aabb.h"
#include "intersection.h"
#include "ray.h"
#include "triangle.h"


namespace Saiga
{
namespace AccelerationStructure
{
using Intersection::RayTriangleIntersection;
/**
 * Base class for triangle acceleration structures.
 */
class SAIGA_CORE_API Base
{
   public:
    virtual ~Base() {}

    virtual RayTriangleIntersection getClosest(const Ray& ray) const          = 0;
    virtual std::vector<RayTriangleIntersection> getAll(const Ray& ray) const = 0;

    float bvh_epsilon      = 0.0001;
    float triangle_epsilon = 0.00001;
};



class SAIGA_CORE_API BruteForce : public Base
{
   public:
    BruteForce(const std::vector<Triangle>& triangles);
    virtual ~BruteForce() {}

    virtual RayTriangleIntersection getClosest(const Ray& ray) const override;
    virtual std::vector<RayTriangleIntersection> getAll(const Ray& ray) const override;

   private:
    std::vector<Triangle> triangles;
};


struct BVHNode
{
    AABB box;

    struct
    {
        uint32_t _inner : 1;
        uint32_t _left : 31;
    };
    uint32_t _right;
};

class SAIGA_CORE_API BVH : public Base
{
   public:
    struct SortTriangleByAxis
    {
        SortTriangleByAxis(int a) : axis(a) {}
        bool operator()(const std::pair<Triangle, int>& A, const std::pair<Triangle, int>& B)
        {
            auto a = A.first.center();
            auto b = B.first.center();
            return a[axis] < b[axis];
        }
        int axis;
    };

    BVH() {}
    BVH(const std::vector<Triangle>& triangles);
    virtual ~BVH() {}

    virtual void construct() = 0;


    virtual RayTriangleIntersection getClosest(const Ray& ray) const override;
    virtual std::vector<RayTriangleIntersection> getAll(const Ray& ray) const override;
    virtual std::pair<float, int> ClosestPoint(const vec3& p) const;

   protected:
    std::vector<std::pair<Triangle, int>> triangles;
    std::vector<BVHNode> nodes;

    AABB computeBox(int start, int end) const;
    void sortByAxis(int start, int end, int axis);

    // Recursive traversal
    void getClosest(int node, const Ray& ray, RayTriangleIntersection& result) const;
    void getAll(int node, const Ray& ray, std::vector<RayTriangleIntersection>& result) const;
    void ClosestPoint(int node, const vec3& p, std::pair<float, int>& result) const;
};

class SAIGA_CORE_API ObjectMedianBVH : public BVH
{
   public:
    ObjectMedianBVH() {}
    ObjectMedianBVH(const std::vector<Triangle>& triangles, int leafTriangles = 5)
        : BVH(triangles), leafTriangles(leafTriangles)
    {
        construct();
    }
    virtual ~ObjectMedianBVH() {}

   protected:
    int leafTriangles;
    void construct() override;
    int construct(int start, int end);
};

}  // namespace AccelerationStructure
}  // namespace Saiga
