/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "aabb.h"

#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include <iostream>



namespace Saiga
{
int AABB::maxDimension() const
{
    vec3 d = max - min;

    float m = -234646;
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

float AABB::maxSize() const
{
    vec3 d = max - min;

    float m = -23462;

    for (int i = 0; i < 3; ++i)
    {
        if (d[i] > m)
        {
            m = d[i];
        }
    }
    return m;
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
    if (min[0] > max[0])
    {
        tmp    = min[0];
        min[0] = max[0];
        max[0] = tmp;
    }

    if (min[1] > max[1])
    {
        tmp    = min[1];
        min[1] = max[1];
        max[1] = tmp;
    }

    if (min[2] > max[2])
    {
        tmp    = min[2];
        min[2] = max[2];
        max[2] = tmp;
    }
}

vec3 AABB::cornerPoint(int cornerIndex) const
{
    SAIGA_ASSERT(0 <= cornerIndex && cornerIndex <= 7);
    switch (cornerIndex)
    {
        default:
        case 0:
            return vec3(min[0], min[1], min[2]);
        case 1:
            return vec3(min[0], min[1], max[2]);
        case 2:
            return vec3(min[0], max[1], max[2]);
        case 3:
            return vec3(min[0], max[1], min[2]);
        case 4:
            return vec3(max[0], min[1], min[2]);
        case 5:
            return vec3(max[0], min[1], max[2]);
        case 6:
            return vec3(max[0], max[1], max[2]);
        case 7:
            return vec3(max[0], max[1], min[2]);
    }
}

bool AABB::contains(const vec3& p) const
{
    if (min[0] > p[0] || max[0] < p[0]) return false;
    if (min[1] > p[1] || max[1] < p[1]) return false;
    if (min[2] > p[2] || max[2] < p[2]) return false;

    return true;  // overlap
}


std::vector<Triangle> AABB::toTriangles() const
{
    std::vector<Triangle> res = {
        // bottom
        Triangle(vec3(min[0], min[1], min[2]), vec3(max[0], min[1], min[2]), vec3(max[0], min[1], max[2])),
        Triangle(vec3(min[0], min[1], min[2]), vec3(max[0], min[1], max[2]), vec3(min[0], min[1], max[2])),

        // top
        Triangle(vec3(min[0], max[1], min[2]), vec3(max[0], max[1], min[2]), vec3(max[0], max[1], max[2])),
        Triangle(vec3(min[0], max[1], min[2]), vec3(max[0], max[1], max[2]), vec3(min[0], max[1], max[2])),


        // left
        Triangle(vec3(min[0], min[1], min[2]), vec3(min[0], min[1], max[2]), vec3(min[0], max[1], max[2])),
        Triangle(vec3(min[0], min[1], min[2]), vec3(min[0], max[1], max[2]), vec3(min[0], max[1], min[2])),

        // right
        Triangle(vec3(max[0], min[1], min[2]), vec3(max[0], min[1], max[2]), vec3(max[0], max[1], max[2])),
        Triangle(vec3(max[0], min[1], min[2]), vec3(max[0], max[1], max[2]), vec3(max[0], max[1], min[2])),


        // back
        Triangle(vec3(min[0], min[1], min[2]), vec3(min[0], max[1], min[2]), vec3(max[0], max[1], min[2])),
        Triangle(vec3(min[0], min[1], min[2]), vec3(max[0], max[1], min[2]), vec3(max[0], min[1], min[2])),


        // front
        Triangle(vec3(min[0], min[1], max[2]), vec3(min[0], max[1], max[2]), vec3(max[0], max[1], max[2])),
        Triangle(vec3(min[0], min[1], max[2]), vec3(max[0], max[1], max[2]), vec3(max[0], min[1], max[2]))};

    return res;
}


std::ostream& operator<<(std::ostream& os, const AABB& bb)
{
    std::cout << "AABB: [" << bb.min.transpose() << "] [" << bb.max.transpose() << "]";
    return os;
}

}  // namespace Saiga
