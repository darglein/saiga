/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/math.h"

#include "aabb.h"

namespace Saiga
{
class SAIGA_GLOBAL Sphere
{
   public:
    vec3 pos;
    float r;


    Sphere(void) {}

    Sphere(const vec3& p, float r) : pos(p), r(r) {}
    ~Sphere(void) {}



    int intersectAabb(const AABB& other);
    bool intersectAabb2(const AABB& other);

    void getMinimumAabb(AABB& box);

    bool contains(vec3 p);
    bool intersect(const Sphere& other);

    //    TriangleMesh* createMesh(int rings, int sectors);
    //    void addToBuffer(std::vector<VertexNT> &vertices, std::vector<GLuint> &indices, int rings, int sectors);

    SAIGA_GLOBAL friend std::ostream& operator<<(std::ostream& os, const Sphere& dt);
};

}  // namespace Saiga
