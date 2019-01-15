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



class SAIGA_GLOBAL BruteForce : Base
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

class SAIGA_GLOBAL BVH : Base
{
   public:
    BVH(const std::vector<Triangle>& triangles);
    virtual ~BVH() {}

    virtual RayTriangleIntersection getClosest(const Ray& ray) override;
    virtual std::vector<RayTriangleIntersection> getAll(const Ray& ray) override;

   private:
    std::vector<Triangle> triangles;
    std::vector<BVHNode> nodes;

    // Recursive traversal
    void getClosest(int node, const Ray& ray, RayTriangleIntersection& result);
};


}  // namespace AccelerationStructure
}  // namespace Saiga
