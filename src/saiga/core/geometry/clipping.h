/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

#include "polygon.h"

namespace Saiga
{
namespace Clipping
{
/**
 * Sutherland–Hodgman algorithm
 *
 * Clips each input edge of the polygon to all clip planes.
 * Only implemented for AABBs here.
 * TODO: More general convex clip objects
 *
 * Some sources:
 * https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
 * https://codereview.stackexchange.com/questions/131852/high-performance-triangle-axis-aligned-bounding-box-clipping
 * https://github.com/matt77hias/Clipping
 */

enum class PlaneSide
{
    POSITIVE,
    NEGATIVE,
    ON_PLANE
};


inline PlaneSide vertexAxisAlignedPlane(vec3 p, int axis, float d, bool isMin)
{
    //    const float PLANE_THICKNESS_EPSILON = 0.000001f;
    float dis = (p[axis] - d);
    dis *= isMin ? 1.0f : -1.0f;
    //       if      (dis >  PLANE_THICKNESS_EPSILON) return PlaneSide::POSITIVE;
    //       else if (dis < -PLANE_THICKNESS_EPSILON) return PlaneSide::NEGATIVE;
    //       else                                   return PlaneSide::ON_PLANE;
    if (dis >= 0)
        return PlaneSide::POSITIVE;
    else
        return PlaneSide::NEGATIVE;
}

inline vec3 intersection(vec3 p1, vec3 p2, int axis, float d)
{
    const double alpha = (d - p1[axis]) / (p2[axis] - p1[axis]);
    vec3 res           = mix(p1, p2, alpha);
    return res;
}

inline PolygonType clipPolygonAxisAlignedPlane(const PolygonType& polygon, int axis, float d, bool isMin)
{
    PolygonType res;
    if (polygon.size() <= 1)
    {
        return res;
    }

    vec3 p1      = polygon.back();
    PlaneSide s1 = vertexAxisAlignedPlane(p1, axis, d, isMin);

    for (vec3 p2 : polygon)
    {
        PlaneSide s2 = vertexAxisAlignedPlane(p2, axis, d, isMin);

        if (s2 == PlaneSide::POSITIVE)
        {
            if (s1 == PlaneSide::NEGATIVE)
            {
                res.push_back(intersection(p1, p2, axis, d));
            }
            res.push_back(p2);
        }
        else
        {
            if (s1 == PlaneSide::POSITIVE)
            {
                res.push_back(intersection(p1, p2, axis, d));
            }
        }
        p1 = p2;
        s1 = s2;
    }
    return res;
}

inline PolygonType clipPolygonAABB(const PolygonType& p, const AABB& box)
{
    PolygonType res = p;
    for (int axis = 0; axis < 3; axis++)
    {
        res = clipPolygonAxisAlignedPlane(res, axis, box.min[axis], true);
        res = clipPolygonAxisAlignedPlane(res, axis, box.max[axis], false);
    }
    return res;
}

inline PolygonType clipTriAABB(const Triangle& tri, const AABB& box)
{
    PolygonType triP = {tri.a, tri.b, tri.c};
    return clipPolygonAABB(triP, box);
}

inline AABB clipTriAABBtoBox(Triangle tri, AABB box)
{
    auto p = clipTriAABB(tri, box);
    return Polygon::boundingBox(p);
}

}  // namespace Clipping

}  // namespace Saiga
