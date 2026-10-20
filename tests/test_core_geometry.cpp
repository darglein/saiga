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

// Regression test: PlanePlane returned the negated line point, because it used the formula for planes of the form
// n * x + d = 0 while Plane stores n * x - d = 0.
TEST(Geometry, PlanePlaneIntersection)
{
    Plane p1(vec3(0, 0, 1), vec3(0, 0, 1));  // z = 1
    Plane p2(vec3(2, 0, 0), vec3(1, 0, 0));  // x = 2
    Ray line;
    ASSERT_TRUE(Intersection::PlanePlane(p1, p2, line));
    EXPECT_NEAR(p1.distance(line.origin), 0, 1e-5);
    EXPECT_NEAR(p2.distance(line.origin), 0, 1e-5);
    EXPECT_NEAR(std::abs(line.direction.normalized().y()), 1, 1e-5);

    // Two general planes: every point on the line lies on both planes.
    Plane p3(vec3(1, 2, 3), vec3(1, 1, 0));
    Plane p4(vec3(-1, 0.5, 2), vec3(0, 1, 2));
    ASSERT_TRUE(Intersection::PlanePlane(p3, p4, line));
    for (float t : {-2.f, 0.f, 3.f})
    {
        vec3 p = line.positionOnRay(t);
        EXPECT_NEAR(p3.distance(p), 0, 1e-4);
        EXPECT_NEAR(p4.distance(p), 0, 1e-4);
    }

    // Parallel planes do not intersect.
    EXPECT_FALSE(Intersection::PlanePlane(p1, Plane(vec3(0, 0, 3), vec3(0, 0, -1)), line));
}

// Regression test: Intersection::SphereAABB was declared and exported, but never defined (link error).
TEST(Geometry, SphereAABBIntersection)
{
    AABB box(vec3(0, 0, 0), vec3(1, 1, 1));
    EXPECT_TRUE(Intersection::SphereAABB(vec3(0.5, 0.5, 0.5), 0.1, box));
    EXPECT_TRUE(Intersection::SphereAABB(vec3(1.5, 0.5, 0.5), 0.6, box));
    EXPECT_FALSE(Intersection::SphereAABB(vec3(1.5, 0.5, 0.5), 0.4, box));
    // Closest point is the corner (1,1,1) at distance sqrt(3) * 0.5.
    EXPECT_TRUE(Intersection::SphereAABB(vec3(1.5, 1.5, 1.5), 0.9, box));
    EXPECT_FALSE(Intersection::SphereAABB(vec3(1.5, 1.5, 1.5), 0.8, box));
}

// Regression test: operator<< of Plane was not exported, so it could not be used outside of saiga_core.
TEST(Geometry, PlaneStreamOperator)
{
    std::stringstream strm;
    strm << Plane(vec3(0, 0, 1), vec3(0, 0, 1));
    EXPECT_NE(strm.str().find("d = 1"), std::string::npos);
}

static void ExpectFrustumMatchesMatrices(Camera& cam)
{
    cam.recalculatePlanes();
    auto vertices = cam.vertices;
    auto planes   = cam.planes;
    cam.recalculatePlanesFromMatrices();
    for (int i = 0; i < 8; ++i)
    {
        EXPECT_NEAR((vertices[i] - cam.vertices[i]).norm(), 0, 1e-3)
            << i << ": " << vertices[i].transpose() << " | " << cam.vertices[i].transpose();
    }
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_NEAR((planes[i].normal - cam.planes[i].normal).norm(), 0, 1e-3) << i;
        EXPECT_NEAR(planes[i].d, cam.planes[i].d, 1e-3) << i;
    }
}

// Regression test: the frustum of a perspective camera with aspect > 1 was too large, because perspective() applies
// fovy to the longer (horizontal) side, but recalculatePlanes() applied it vertically.
TEST(Camera, PerspectiveFrustumMatchesProjection)
{
    for (float aspect : {0.5f, 1.0f, 2.0f})
    {
        PerspectiveCamera cam;
        cam.setProj(60, aspect, 0.5, 20);
        cam.setView(vec3(1, 2, 3), vec3(0, 0, -5), vec3(0, 1, 0));
        ExpectFrustumMatchesMatrices(cam);
    }
}

// Regression test: OrthographicCamera::recalculatePlanes used the up vector as viewing direction.
TEST(Camera, OrthographicFrustumMatchesProjection)
{
    OrthographicCamera cam;
    cam.setProj(-2, 3, -1, 4, 0.5, 20);
    cam.setView(vec3(1, 2, 3), vec3(0, 0, -5), vec3(0, 1, 0));
    ExpectFrustumMatchesMatrices(cam);

    cam.setView(vec3(0, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0));
    cam.recalculatePlanes();
    EXPECT_EQ(cam.sphereInFrustum(Sphere(vec3(0, 1, -10), 0.1)), Frustum::INSIDE);
    EXPECT_EQ(cam.sphereInFrustum(Sphere(vec3(0, 1, 10), 0.1)), Frustum::OUTSIDE);
    EXPECT_EQ(cam.sphereInFrustum(Sphere(vec3(0, -10, -10), 0.1)), Frustum::OUTSIDE);
}

// Regression test: the operator<< of Camera was not exported, so it could not be used outside of saiga_core.
TEST(Camera, StreamOperator)
{
    PerspectiveCamera cam;
    cam.name = "my_camera";
    std::stringstream strm;
    strm << static_cast<const Camera&>(cam);
    EXPECT_EQ(strm.str(), "my_camera");
}

TEST(Camera, ProjectUnproject)
{
    PerspectiveCamera cam;
    cam.setProj(70, 1.5, 0.1, 100);
    cam.setView(vec3(1, 2, 3), vec3(0, 0, -5), vec3(0, 1, 0));

    vec3 world(0.5, 0.2, -4);
    vec3 view = cam.WorldToView(world);
    EXPECT_LT(view.z(), 0);
    vec3 ndc = cam.ViewToNormalized(view);
    vec2 ip  = cam.NormalizedToImage(ndc, 640, 480);

    vec3 ndc2   = cam.ImageToNormalized(ip, ndc.z(), 640, 480);
    vec3 world2 = cam.ViewToWorld(cam.NormalizedToView(ndc2));
    EXPECT_NEAR((world - world2).norm(), 0, 1e-3);

    // The pixel ray through the projected point passes through the world point.
    Ray r = cam.PixelRay(ip, 640, 480, false);
    EXPECT_NEAR(r.direction.norm(), 1, 1e-5);
    vec3 to_point = (world - r.origin).normalized();
    EXPECT_NEAR(to_point.dot(r.direction), 1, 1e-5);

    // Depth linearization round trip.
    float d = 0.7f;
    EXPECT_NEAR(cam.nonlinearDepth(cam.linearDepth(d)), d, 1e-5);
}

TEST(Camera, CVIntrinsicsRoundTrip)
{
    mat3 K  = mat3::Identity();
    K(0, 0) = 500;
    K(1, 1) = 520;
    K(0, 2) = 310;
    K(1, 2) = 250;
    mat4 P  = CVCamera2GLProjectionMatrix(K, ivec2(640, 480), 0.1, 100);
    mat3 K2 = GLProjectionMatrix2CVCamera(P, 640, 480);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(K(i, j), K2(i, j), 1e-3) << i << " " << j;
        }
    }

    // A point in CV camera coordinates projects to the same pixel with K and with the GL projection.
    vec3 p_cv(0.3, -0.2, 2.0);
    vec3 pix = K * p_cv;
    pix /= pix.z();
    vec4 p_gl  = GL2CVView() * make_vec4(p_cv, 1);
    vec4 clip  = P * p_gl;
    vec3 ndc   = make_vec3(clip) / clip.w();
    float u_gl = (ndc.x() * 0.5f + 0.5f) * 640;
    float v_gl = (1 - (ndc.y() * 0.5f + 0.5f)) * 480;
    EXPECT_NEAR(pix.x(), u_gl, 1e-2);
    EXPECT_NEAR(pix.y(), v_gl, 1e-2);
}
