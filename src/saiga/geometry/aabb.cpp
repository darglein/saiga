/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/geometry/aabb.h"

#include "saiga/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include <glm/gtc/epsilon.hpp>

namespace Saiga
{
int AABB::maxDimension()
{
    vec3 d = max - min;

    float m = -1;
    int mi  = -1;

    for (int i = 0; i < 3; ++i)
    {
        if (d[i] > m)
        {
            mi = i;
            m  = d[i];
        }
    }
    return mi;
}

#define MIN(X, Y) ((X < Y) ? X : Y)
#define MAX(X, Y) ((X > Y) ? X : Y)
#define MINV(V1, V2) vec3(MIN(V1[0], V2[0]), MIN(V1[1], V2[1]), MIN(V1[2], V2[2]))
#define MAXV(V1, V2) vec3(MAX(V1[0], V2[0]), MAX(V1[1], V2[1]), MAX(V1[2], V2[2]))
void AABB::growBox(const vec3& v)
{
    min = MINV(min, v);
    max = MAXV(max, v);
}

void AABB::growBox(const AABB& v)
{
    min = MINV(min, v.min);
    max = MAXV(max, v.max);
}



void AABB::ensureValidity()
{
    float tmp;
    if (min.x > max.x)
    {
        tmp   = min.x;
        min.x = max.x;
        max.x = tmp;
    }

    if (min.y > max.y)
    {
        tmp   = min.y;
        min.y = max.y;
        max.y = tmp;
    }

    if (min.z > max.z)
    {
        tmp   = min.z;
        min.z = max.z;
        max.z = tmp;
    }
}

vec3 AABB::cornerPoint(int cornerIndex) const
{
    SAIGA_ASSERT(0 <= cornerIndex && cornerIndex <= 7);
    switch (cornerIndex)
    {
        default:
        case 0:
            return vec3(min.x, min.y, min.z);
        case 1:
            return vec3(min.x, min.y, max.z);
        case 2:
            return vec3(min.x, max.y, max.z);
        case 3:
            return vec3(min.x, max.y, min.z);
        case 4:
            return vec3(max.x, min.y, min.z);
        case 5:
            return vec3(max.x, min.y, max.z);
        case 6:
            return vec3(max.x, max.y, max.z);
        case 7:
            return vec3(max.x, max.y, min.z);
    }
}

bool AABB::contains(const vec3& p)
{
    if (min.x > p.x || max.x < p.x) return false;
    if (min.y > p.y || max.y < p.y) return false;
    if (min.z > p.z || max.z < p.z) return false;

    return true;  // overlap
}

std::vector<Triangle> AABB::toTriangles()
{
    std::vector<Triangle> res = {
        // bottom
        Triangle(vec3(min.x, min.y, min.z), vec3(max.x, min.y, min.z), vec3(max.x, min.y, max.z)),
        Triangle(vec3(min.x, min.y, min.z), vec3(max.x, min.y, max.z), vec3(min.x, min.y, max.z)),

        // top
        Triangle(vec3(min.x, max.y, min.z), vec3(max.x, max.y, min.z), vec3(max.x, max.y, max.z)),
        Triangle(vec3(min.x, max.y, min.z), vec3(max.x, max.y, max.z), vec3(min.x, max.y, max.z)),


        // left
        Triangle(vec3(min.x, min.y, min.z), vec3(min.x, min.y, max.z), vec3(min.x, max.y, max.z)),
        Triangle(vec3(min.x, min.y, min.z), vec3(min.x, max.y, max.z), vec3(min.x, max.y, min.z)),

        // right
        Triangle(vec3(max.x, min.y, min.z), vec3(max.x, min.y, max.z), vec3(max.x, max.y, max.z)),
        Triangle(vec3(max.x, min.y, min.z), vec3(max.x, max.y, max.z), vec3(max.x, max.y, min.z)),


        // back
        Triangle(vec3(min.x, min.y, min.z), vec3(min.x, max.y, min.z), vec3(max.x, max.y, min.z)),
        Triangle(vec3(min.x, min.y, min.z), vec3(max.x, max.y, min.z), vec3(max.x, min.y, min.z)),


        // front
        Triangle(vec3(min.x, min.y, max.z), vec3(min.x, max.y, max.z), vec3(max.x, max.y, max.z)),
        Triangle(vec3(min.x, min.y, max.z), vec3(max.x, max.y, max.z), vec3(max.x, min.y, max.z))};

    return res;
}


std::ostream& operator<<(std::ostream& os, const AABB& bb)
{
    sampleCone(bb.min, 5);
    std::cout << "AABB: " << bb.min << " " << bb.max;
    return os;
}

}  // namespace Saiga
