/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/geometry/aabb.h"
#include "saiga/geometry/intersection.h"
#include "saiga/geometry/ray.h"
#include "saiga/geometry/triangle.h"
#include "saiga/util/math.h"


namespace Saiga
{
namespace AccelerationStructure
{
using Intersection::RayTriangleIntersection;
/**
 * Base class for triangle acceleration structures.
 */
class SAIGA_GLOBAL Base
{
   public:
    virtual ~Base() {}

    virtual RayTriangleIntersection getClosest(const Ray& ray)          = 0;
    virtual std::vector<RayTriangleIntersection> getAll(const Ray& ray) = 0;
};



class SAIGA_GLOBAL BruteForce : public Base
{
   public:
    BruteForce(const std::vector<Triangle>& triangles);
    virtual ~BruteForce() {}

    virtual RayTriangleIntersection getClosest(const Ray& ray) override;
    virtual std::vector<RayTriangleIntersection> getAll(const Ray& ray) override;

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

class SAIGA_GLOBAL BVH : public Base
{
   public:
    struct SortTriangleByAxis
    {
        SortTriangleByAxis(int a) : axis(a) {}
        bool operator()(const Triangle& A, const Triangle& B)
        {
            auto a = A.center();
            auto b = B.center();
            return a[axis] < b[axis];
        }
        int axis;
    };

    BVH(const std::vector<Triangle>& triangles);
    virtual ~BVH() {}

    virtual void construct() = 0;


    virtual RayTriangleIntersection getClosest(const Ray& ray) override;
    virtual std::vector<RayTriangleIntersection> getAll(const Ray& ray) override;

   protected:
    std::vector<Triangle> triangles;
    std::vector<BVHNode> nodes;

    AABB computeBox(int start, int end);
    void sortByAxis(int start, int end, int axis);

    // Recursive traversal
    void getClosest(int node, const Ray& ray, RayTriangleIntersection& result);
    void getAll(int node, const Ray& ray, std::vector<RayTriangleIntersection>& result);
};

class SAIGA_GLOBAL ObjectMedianBVH : public BVH
{
   public:
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
