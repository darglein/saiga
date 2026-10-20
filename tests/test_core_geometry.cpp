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

static void ExpectMatNear(const mat4& a, const mat4& b, float eps)
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            EXPECT_NEAR(a(i, j), b(i, j), eps) << i << " " << j;
        }
    }
}

// Regression test: turn() should not rotate the object upside down. It checked the up vector of the stale 'model'
// matrix and restored only 'model', so the rotation was never limited.
TEST(Object3D, TurnDoesNotFlipOver)
{
    Object3D obj;
    obj.calculateModel();
    for (int i = 0; i < 20; ++i)
    {
        obj.turn(0, 20, vec3(0, 1, 0));
        obj.calculateModel();
        EXPECT_GE(obj.getUpVector().y(), -1e-5) << i;
    }
    // Turning sideways is not limited.
    obj.turn(30, 0, vec3(0, 1, 0));
    EXPECT_GE(obj.getUpVector().y(), -1e-5);
}

// Regression test: rotateAroundPoint rotated the position around the global axis but the orientation around the
// local axis, so a rotated object ended up with a wrong orientation.
TEST(Object3D, RotateAroundPoint)
{
    Object3D obj;
    obj.setPosition(vec3(1, 0, 0));
    obj.rot     = angleAxis(radians(90.f), vec3(0, 0, 1));
    mat4 before = obj.getModelMatrix();

    vec3 point(0, 0, 2);
    obj.rotateAroundPoint(point, vec3(1, 0, 0), 90);

    mat4 expected = translate(point) * rotate(radians(90.f), vec3(1, 0, 0)) * translate(-point) * before;
    ExpectMatNear(obj.getModelMatrix(), expected, 1e-5);
}

// Regression test: getSimpleDirectionQuat built a non-orthonormal matrix if 'dir' was not horizontal, so the
// resulting rotation did not look along 'dir'.
TEST(Object3D, SimpleDirection)
{
    for (vec3 dir : {vec3(0, 0, -1), vec3(1, 0, 0), vec3(1, 1, 0), vec3(-2, -1, 3), vec3(0.1, -5, 0.3)})
    {
        Object3D obj;
        obj.setSimpleDirection(dir);
        EXPECT_NEAR(obj.rot.norm(), 1, 1e-5);
        // The object looks along -z, like an OpenGL camera.
        vec3 look = -make_vec3(obj.getDirection());
        EXPECT_NEAR((look - dir.normalized()).norm(), 0, 1e-5) << dir.transpose();
        // The right vector is horizontal.
        EXPECT_NEAR(obj.getRightVector().y(), 0, 1e-5) << dir.transpose();
        EXPECT_GT(obj.getUpVector().y(), 0) << dir.transpose();
    }
}

TEST(Object3D, ModelMatrixRoundTrip)
{
    Object3D obj;
    obj.setPosition(vec3(1, -2, 3));
    obj.rot = angleAxis(0.7f, vec3(1, 2, 3).normalized());
    obj.setScale(vec3(2, 0.5, 3));
    mat4 m = obj.getModelMatrix();

    Object3D obj2;
    obj2.setModelMatrix(m);
    ExpectMatNear(obj2.getModelMatrix(), m, 1e-5);
    EXPECT_NEAR((obj2.getScale() - vec3(2, 0.5, 3)).norm(), 0, 1e-5);

    // Global and local rotations.
    Object3D a;
    a.rot      = angleAxis(0.3f, vec3(0, 1, 0));
    Object3D b = a;
    a.rotateGlobal(vec3(1, 0, 0), 40);
    b.rotateLocal(vec3(1, 0, 0), 40);
    ExpectMatNear(make_mat4(a.rot), rotate(radians(40.f), vec3(1, 0, 0)) * make_mat4(angleAxis(0.3f, vec3(0, 1, 0))),
                  1e-5);
    ExpectMatNear(make_mat4(b.rot), make_mat4(angleAxis(0.3f, vec3(0, 1, 0))) * rotate(radians(40.f), vec3(1, 0, 0)),
                  1e-5);

    // Interpolation hits both end points.
    Object3D i0 = Object3D::interpolate(obj, a, 0);
    Object3D i1 = Object3D::interpolate(obj, a, 1);
    ExpectMatNear(i0.getModelMatrix(), obj.getModelMatrix(), 1e-5);
    ExpectMatNear(i1.getModelMatrix(), a.getModelMatrix(), 1e-5);
}

// Regression test: isDegenerate missed triangles with two identical corners (the corner angles are NaN) and exactly
// collinear corners (rounding gives cos(angle) slightly below 1).
TEST(Geometry, TriangleDegenerate)
{
    vec3 a(0.1, 0.7, 0.3), b(1, 2, 3.3), c(-0.4, 1.1, 2);
    EXPECT_FALSE(Triangle(a, b, c).isDegenerate());
    EXPECT_TRUE(Triangle(a, a, c).isDegenerate());
    EXPECT_TRUE(Triangle(a, b, b).isDegenerate());
    EXPECT_TRUE(Triangle(c, b, c).isDegenerate());
    EXPECT_TRUE(Triangle(a, a, a).isDegenerate());
    EXPECT_TRUE(Triangle(vec3(0, 0, 0), vec3(1, 2, 3), vec3(2, 4, 6)).isDegenerate());
    EXPECT_TRUE(Triangle(vec3(1, 1, 1), vec3(3, 3, 3), vec3(2, 2, 2)).isDegenerate());
}

// Regression test: Sphere::getMinimumAabb added 1 to the radius, so the box was not the minimal one.
TEST(Geometry, SphereMinimumAabb)
{
    Sphere s(vec3(1, 2, 3), 0.5);
    AABB box;
    s.getMinimumAabb(box);
    EXPECT_NEAR((box.min - vec3(0.5, 1.5, 2.5)).norm(), 0, 1e-6);
    EXPECT_NEAR((box.max - vec3(1.5, 2.5, 3.5)).norm(), 0, 1e-6);
}

TEST(Geometry, TriangleFunctions)
{
    Triangle t(vec3(0, 0, 0), vec3(2, 0, 0), vec3(0, 2, 0));
    EXPECT_NEAR(t.Area(), 2, 1e-6);
    EXPECT_NEAR((t.normal() - vec3(0, 0, 1)).norm(), 0, 1e-6);
    EXPECT_NEAR((t.center() - vec3(2.f / 3, 2.f / 3, 0)).norm(), 0, 1e-6);
    EXPECT_NEAR(t.angleAtCorner(0), pi<float>() / 2, 1e-5);
    EXPECT_NEAR(t.angleAtCorner(1), pi<float>() / 4, 1e-5);
    EXPECT_NEAR(t.minimalAngle(), pi<float>() / 4, 1e-5);

    // Distance to the interior, an edge and a corner.
    EXPECT_NEAR(t.Distance(vec3(0.5, 0.5, 3)), 3, 1e-5);
    EXPECT_NEAR(t.Distance(vec3(1, -1, 0)), 1, 1e-5);
    EXPECT_NEAR(t.Distance(vec3(3, -1, 0)), std::sqrt(2.f), 1e-5);
    EXPECT_NEAR(t.Distance(vec3(-1, -1, 0)), std::sqrt(2.f), 1e-5);
    EXPECT_NEAR(t.Distance(vec3(2, 2, 0)), std::sqrt(2.f), 1e-5);

    // Barycentric coordinates reproduce the projected point.
    vec3 p(0.5, 0.25, 7);
    vec3 bary = t.BarycentricCoordinates(p);
    EXPECT_NEAR(bary.sum(), 1, 1e-6);
    EXPECT_NEAR((t.InterpolateBarycentric(bary) - vec3(0.5, 0.25, 0)).norm(), 0, 1e-5);

    Random::setSeed(10);
    for (int i = 0; i < 100; ++i)
    {
        vec3 r = t.RandomPointOnSurface();
        EXPECT_NEAR(t.Distance(r), 0, 1e-5);
        vec3 b = t.RandomBarycentric();
        EXPECT_NEAR(b.sum(), 1, 1e-5);
        EXPECT_GE(b.minCoeff(), 0);
    }

    Triangle t2 = t;
    t2.ScaleUniform(2);
    EXPECT_NEAR(t2.Area(), 8, 1e-5);
    EXPECT_NEAR((t2.center() - t.center()).norm(), 0, 1e-6);
}

// Regression test: a nearest neighbor search on an empty tree read an uninitialized node index and a k-nearest
// neighbor search with k = 0 accessed queue[-1].
TEST(Geometry, KDTreeEmptyQueries)
{
    KDTree<3, vec3> empty_tree(std::vector<vec3>{});
    EXPECT_EQ(empty_tree.NearestNeighborSearch(vec3(0, 0, 0)), -1);
    EXPECT_TRUE(empty_tree.KNearestNeighborSearch(vec3(0, 0, 0), 3).empty());
    EXPECT_TRUE(empty_tree.RadiusSearch(vec3(0, 0, 0), 1).empty());

    KDTree<3, vec3> tree(std::vector<vec3>{vec3(0, 0, 0), vec3(1, 0, 0)});
    EXPECT_TRUE(tree.KNearestNeighborSearch(vec3(0, 0, 0), 0).empty());
    EXPECT_EQ(tree.NearestNeighborSearch(vec3(0.9, 0, 0)), 1);
}

static std::vector<int> SortedNeighbours(HalfEdgeMesh<Vertex, uint32_t>& hem, int v)
{
    std::vector<int> n;
    hem.getNeighbours(v, n);
    std::sort(n.begin(), n.end());
    return n;
}

// Regression test: getNeighbours missed the last neighbor of a boundary vertex when walking in the second direction
// (e.g. vertex 0 of a single triangle only returned one neighbor).
TEST(Geometry, HalfEdgeMeshNeighbours)
{
    TriangleMesh<Vertex, uint32_t> mesh;
    // A square of 4 triangles around the center vertex 4.
    mesh.addVertex(Vertex(vec3(0, 0, 0)));
    mesh.addVertex(Vertex(vec3(1, 0, 0)));
    mesh.addVertex(Vertex(vec3(1, 1, 0)));
    mesh.addVertex(Vertex(vec3(0, 1, 0)));
    mesh.addVertex(Vertex(vec3(0.5, 0.5, 0)));
    mesh.addFace(0, 1, 4);
    mesh.addFace(1, 2, 4);
    mesh.addFace(2, 3, 4);
    mesh.addFace(3, 0, 4);

    HalfEdgeMesh<Vertex, uint32_t> hem(mesh);
    EXPECT_TRUE(hem.isValid());
    EXPECT_EQ(SortedNeighbours(hem, 4), std::vector<int>({0, 1, 2, 3}));
    for (int v = 0; v < 4; ++v)
    {
        std::vector<int> expected = {(v + 1) % 4, (v + 3) % 4, 4};
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(SortedNeighbours(hem, v), expected) << v;
    }

    TriangleMesh<Vertex, uint32_t> single;
    single.addVertex(Vertex(vec3(0, 0, 0)));
    single.addVertex(Vertex(vec3(1, 0, 0)));
    single.addVertex(Vertex(vec3(0, 1, 0)));
    single.addFace(0, 1, 2);
    HalfEdgeMesh<Vertex, uint32_t> hem2(single);
    EXPECT_EQ(SortedNeighbours(hem2, 0), std::vector<int>({1, 2}));
    EXPECT_EQ(SortedNeighbours(hem2, 1), std::vector<int>({0, 2}));
    EXPECT_EQ(SortedNeighbours(hem2, 2), std::vector<int>({0, 1}));

    // Round trip back to an indexed face set.
    TriangleMesh<Vertex, uint32_t> out;
    hem.toIFS(out);
    EXPECT_EQ(out.faces.size(), mesh.faces.size());
    EXPECT_EQ(out.vertices.size(), mesh.vertices.size());
    EXPECT_EQ(out.toTriangleList().size(), 4u);
}

// Regression test: closestPointOnCircle normalized a zero vector (NaN) if the point projects onto the circle center.
TEST(Geometry, CircleClosestPoint)
{
    Circle c(vec3(1, 2, 3), 2, vec3(0, 0, 1));
    vec3 p = c.closestPointOnCircle(vec3(4, 2, 7));
    EXPECT_NEAR((p - vec3(3, 2, 3)).norm(), 0, 1e-5);
    EXPECT_NEAR(c.distance(vec3(4, 2, 7)), std::sqrt(1.f + 16.f), 1e-5);

    // Every point on the circle is closest to a point on the axis.
    vec3 q = c.closestPointOnCircle(vec3(1, 2, 10));
    EXPECT_TRUE(q.allFinite());
    EXPECT_NEAR((q - c.pos).norm(), 2, 1e-5);
    EXPECT_NEAR((q - c.pos).dot(c.normal), 0, 1e-5);
    EXPECT_NEAR(c.distance(vec3(1, 2, 10)), std::sqrt(4.f + 49.f), 1e-5);
}
