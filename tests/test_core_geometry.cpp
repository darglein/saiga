/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/camera/camera.h"
#include "saiga/core/geometry/AccelerationStructure.h"
#include "saiga/core/geometry/Frustum.h"
#include "saiga/core/geometry/aabb.h"
#include "saiga/core/geometry/bezier.h"
#include "saiga/core/geometry/bspline.h"
#include "saiga/core/geometry/circle.h"
#include "saiga/core/geometry/clipping.h"
#include "saiga/core/geometry/half_edge_mesh.h"
#include "saiga/core/geometry/intersection.h"
#include "saiga/core/geometry/kdtree.h"
#include "saiga/core/geometry/object3d.h"
#include "saiga/core/geometry/triangle.h"
#include "saiga/core/math/CoordinateSystems.h"
#include "saiga/core/math/random.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <sstream>

using namespace Saiga;

// Regression test: operator<< of AABB and Bspline wrote (parts of) the output to std::cout instead of the given stream.
TEST(Geometry, StreamOperatorsWriteToStream)
{
    std::stringstream strm;
    strm << AABB(vec3(0, 1, 2), vec3(3, 4, 5));
    EXPECT_NE(strm.str().find("AABB"), std::string::npos);

    std::stringstream strm2;
    Bspline<vec2, float> spline({vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)}, 3);
    strm2 << spline;
    EXPECT_NE(strm2.str().find("knots:"), std::string::npos);
    EXPECT_NE(strm2.str().find("control points:"), std::string::npos);
}

// Regression test: pointInFrustum returned OUTSIDE for points inside the frustum and INSIDE for some points outside,
// because the frustum planes point outwards but the test treated a negative distance as outside.
TEST(Geometry, FrustumPointInFrustum)
{
    // Camera at the origin looking along -z.
    Frustum frustum(mat4::Identity(), radians(60.f), 1, 1, 10);

    EXPECT_EQ(frustum.pointInFrustum(vec3(0, 0, -5)), Frustum::INSIDE);
    EXPECT_EQ(frustum.pointInFrustum(vec3(0.5, -0.5, -2)), Frustum::INSIDE);
    EXPECT_EQ(frustum.pointInFrustum(vec3(0, 0, 5)), Frustum::OUTSIDE);
    EXPECT_EQ(frustum.pointInFrustum(vec3(0, 0, -0.5)), Frustum::OUTSIDE);
    EXPECT_EQ(frustum.pointInFrustum(vec3(0, 0, -11)), Frustum::OUTSIDE);
    EXPECT_EQ(frustum.pointInFrustum(vec3(10, 0, -5)), Frustum::OUTSIDE);
    EXPECT_EQ(frustum.pointInFrustum(vec3(0, -10, -5)), Frustum::OUTSIDE);

    // The sphere test agrees with the point test.
    EXPECT_EQ(frustum.sphereInFrustum(Sphere(vec3(0, 0, -5), 0.1)), Frustum::INSIDE);
    EXPECT_EQ(frustum.sphereInFrustum(Sphere(vec3(0, 0, 5), 0.1)), Frustum::OUTSIDE);
    EXPECT_EQ(frustum.sphereInFrustum(Sphere(vec3(0, 0, -1), 0.5)), Frustum::INTERSECT);
}
