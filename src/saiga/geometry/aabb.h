/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/geometry/triangle.h"
#include "saiga/util/math.h"

#include <vector>

namespace Saiga
{
class SAIGA_GLOBAL AABB
{
   public:
    vec3 min = make_vec3(0);
    vec3 max = make_vec3(0);

   public:
    AABB();
    AABB(const vec3& min, const vec3& max);
    ~AABB();


    // returns the axis with the maximum extend
    int maxDimension();

    void makeNegative();
    void growBox(const vec3& v);
    void growBox(const AABB& v);

    void transform(const mat4& trafo);
    void translate(const vec3& v);
    void scale(const vec3& s);
    void ensureValidity();


    vec3 getHalfExtends();



    void getMinimumAabb(AABB& box) { box = *this; }

    vec3 cornerPoint(int i) const;

    vec3 getPosition() const;
    void setPosition(const vec3& v);

    bool contains(const vec3& p);

    // A list the 12 triangles (2 for each face)
    std::vector<Triangle> toTriangles();


    SAIGA_GLOBAL friend std::ostream& operator<<(std::ostream& os, const AABB& dt);
};


}  // namespace Saiga

#include "saiga/geometry/aabb.inl"
