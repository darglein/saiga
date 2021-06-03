/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

#include "triangle.h"

#include <vector>

namespace Saiga
{
// This class can be used in CUDA, however not all member functions are available there.
class SAIGA_CORE_API AABB
{
   public:
    vec3 min = vec3(0, 0, 0);
    vec3 max = vec3(0, 0, 0);

   public:
    HD AABB() {}
    HD AABB(const vec3& min, const vec3& max) : min(min), max(max) {}

    vec3 getPosition() const;
    void setPosition(const vec3& v);
    // returns the axis with the maximum extend
    int maxDimension() const;

    float maxSize() const;

    void makeNegative();

    void transform(const mat4& trafo);
    void translate(const vec3& v);
    void scale(const vec3& s);


    vec3 getHalfExtends() const { return 0.5f * (max - min); }
    vec3 Size() const { return (max - min); }


    // Point to AABB distance
    float DistanceSquared(const vec3& p) const;

    // ================== Defined in aabb.cpp ==================
    // These function are currently not usable in cuda.
    // =============================================================

    void growBox(const vec3& v);
    void growBox(const AABB& v);
    void ensureValidity();
    void getMinimumAabb(AABB& box) const { box = *this; }

    vec3 cornerPoint(int i) const;

    bool contains(const vec3& p) const;


    // A list the 12 triangles (2 for each face)
    std::vector<Triangle> toTriangles() const;

    std::pair<vec3, float> BoundingSphere() const;


    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const AABB& dt);
};


}  // namespace Saiga

#include "saiga/core/geometry/aabb.inl"
