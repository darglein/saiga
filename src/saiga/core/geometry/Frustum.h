/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/geometry/plane.h"
#include "saiga/core/geometry/sphere.h"
#include "saiga/core/math/math.h"
#include <array>

namespace Saiga
{
// This class defines a camera frustum, which is arbitraly rotated truncated pyramid with rectangular base. Such a
// frustum is uniquely defined by the 8 corner points or 6 planes.
class SAIGA_CORE_API Frustum
{
   public:
    enum IntersectionResult
    {
        OUTSIDE = 0,
        INSIDE,
        INTERSECT
    };

    // corners of the truncated pyramid
    // Ordered like this:
    //
    // Near Plane:  Far Plane:
    // 0 -- 1       4 -- 5
    // |    |       |    |
    // 2 -- 3       6 -- 7
    //
    //
    //
    //
    //
    std::array<vec3, 8> vertices;

    // Ordered like this:
    // near, far, top, bottom, left, right
    std::array<Plane, 6> planes;

    Sphere boundingSphere;  // for fast frustum culling


    Frustum() {}

    Frustum(const mat4& model, float fovy, float aspect, float zNear, float zFar, bool negativ_z = true,
            bool negative_y = false);

    void computePlanesFromVertices();

    // culling stuff
    IntersectionResult pointInFrustum(const vec3& p) const;
    IntersectionResult sphereInFrustum(const Sphere& s) const;

    IntersectionResult pointInSphereFrustum(const vec3& p) const;
    IntersectionResult sphereInSphereFrustum(const Sphere& s) const;


    std::array<Triangle, 12> ToTriangleList() const;

    /**
     * Return the intervall (min,max) when all vertices of the frustum are
     * projected to the axis 'd'. To dedect an overlap in intervalls the axis
     * does not have to be normalized.
     *
     * @brief projectedIntervall
     * @param d
     * @return
     */
    vec2 projectedIntervall(const vec3& d) const;

    /**
     * Returns the side of the plane on which the frustum is.
     * +1 on the positive side
     * -1 on the negative side
     * 0 the plane is intersecting the frustum
     *
     * @brief sideOfPlane
     * @param plane
     * @return
     */
    int sideOfPlane(const Plane& plane) const;


    /**
     * Returns unique edges of the frustum.
     * A frustum has 6 unique edges ( non parallel edges).
     * @brief getEdge
     * @param i has to be in range (0 ... 5)
     * @return
     */

    std::pair<vec3, vec3> getEdge(int i) const;

    /**
     * Exact frustum-frustum intersection with the Separating Axes Theorem (SAT).
     * This test is expensive, so it should be only used when important.
     *
     * Number of Operations:
     * 6+6=12  sideOfPlane(const Plane &plane), for testing the faces of the frustum.
     * 6*6*2=72  projectedIntervall(const vec3 &d), for testing all cross product of pairs of non parallel edges
     *
     * http://www.geometrictools.com/Documentation/MethodOfSeparatingAxes.pdf
     * @brief intersectSAT
     * @param other
     * @return
     */

    bool intersectSAT(const Frustum& other) const;



    bool intersectSAT(const Sphere& s) const;
};

SAIGA_CORE_API std::ostream& operator<<(std::ostream& os, const Frustum& frustum);
}  // namespace Saiga
