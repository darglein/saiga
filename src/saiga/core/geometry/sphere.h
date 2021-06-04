/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

#include "aabb.h"

namespace Saiga
{
class SAIGA_CORE_API Sphere
{
   public:
    vec3 pos;
    float r;


    HD Sphere() { static_assert(sizeof(Sphere) == 16, "Sphere must be 16 byte"); }
    HD Sphere(const vec3& p, float r) : pos(p), r(r) {}



    int intersectAabb(const AABB& other) const;
    bool intersectAabb2(const AABB& other) const;

    void getMinimumAabb(AABB& box) const;

    bool contains(vec3 p) const;
    bool intersect(const Sphere& other) const;

    // Signed distance to sphere surface.
    // >0 means outside
    // <0 means inside
    // =0 on the surface
    float sdf(vec3 p) const;


    vec2 projectedIntervall(const vec3& d) const;

    //    TriangleMesh* createMesh(int rings, int sectors);
    //    void addToBuffer(std::vector<VertexNT> &vertices, std::vector<GLuint> &indices, int rings, int sectors);

    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const Sphere& dt);
};

}  // namespace Saiga
