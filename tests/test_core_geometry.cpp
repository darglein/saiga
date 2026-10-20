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
#include "saiga/core/geometry/triangle_mesh.h"
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

TEST(Geometry, RayIntersections)
{
    // Ray - sphere
    float t1, t2;
    ASSERT_TRUE(Intersection::RaySphere(Ray(vec3(0, 0, 1), vec3(0, 0, -5)), Sphere(vec3(0, 0, 0), 2), t1, t2));
    EXPECT_NEAR(t1, 3, 1e-5);
    EXPECT_NEAR(t2, 7, 1e-5);
    EXPECT_FALSE(Intersection::RaySphere(Ray(vec3(0, 0, 1), vec3(3, 0, -5)), Sphere(vec3(0, 0, 0), 2), t1, t2));

    ASSERT_TRUE(Intersection::RayCircle(vec2(-5, 0), vec2(2, 0), vec2(0, 0), 1, t1, t2));
    EXPECT_NEAR(t1, 2, 1e-5);
    EXPECT_NEAR(t2, 3, 1e-5);

    // Ray - triangle. The front side is the counter clockwise side (+z here).
    Triangle tri(vec3(-1, -1, 0), vec3(1, -1, 0), vec3(0, 1, 0));
    auto hit = Intersection::RayTriangle(Ray(vec3(0, 0, -1), vec3(0, 0, 3)), tri);
    ASSERT_TRUE(hit.valid);
    EXPECT_NEAR(hit.t, 3, 1e-5);
    EXPECT_FALSE(hit.backFace);
    hit = Intersection::RayTriangle(Ray(vec3(0, 0, 1), vec3(0, 0, -3)), tri);
    ASSERT_TRUE(hit.valid);
    EXPECT_TRUE(hit.backFace);
    EXPECT_FALSE(Intersection::RayTriangle(Ray(vec3(0, 0, -1), vec3(5, 0, 3)), tri).valid);
    // The triangle is behind the ray
    EXPECT_FALSE(Intersection::RayTriangle(Ray(vec3(0, 0, 1), vec3(0, 0, 3)), tri).valid);

    // Ray - plane
    float t;
    EXPECT_TRUE(Intersection::RayPlane(Ray(vec3(0, -1, 0), vec3(0, 5, 0)), Plane(vec3(0, 2, 0), vec3(0, 1, 0)), t));
    EXPECT_NEAR(t, 3, 1e-5);
    EXPECT_FALSE(Intersection::RayPlane(Ray(vec3(1, 0, 0), vec3(0, 5, 0)), Plane(vec3(0, 2, 0), vec3(0, 1, 0)), t));

    // Ray - AABB
    AABB box(vec3(-1, -1, -1), vec3(1, 1, 1));
    EXPECT_TRUE(Intersection::RayAABB(Ray(vec3(1, 0.5, 0.2).normalized(), vec3(-5, -2.5, -1)), box, t));
    EXPECT_TRUE(Intersection::RayAABB(Ray(vec3(1, 0, 0), vec3(-5, 0.5, 0.5)), box, t));
    EXPECT_NEAR(t, 4, 1e-5);
    EXPECT_FALSE(Intersection::RayAABB(Ray(vec3(-1, 0, 0), vec3(-5, 0.5, 0.5)), box, t));
    EXPECT_FALSE(Intersection::RayAABB(Ray(vec3(1, 0, 0), vec3(-5, 2.5, 0.5)), box, t));

    EXPECT_TRUE(Intersection::SphereSphere(Sphere(vec3(0, 0, 0), 1), Sphere(vec3(1.5, 0, 0), 1)));
    EXPECT_FALSE(Intersection::SphereSphere(Sphere(vec3(0, 0, 0), 1), Sphere(vec3(2.5, 0, 0), 1)));
    EXPECT_TRUE(Intersection::AABBAABB(box, AABB(vec3(0.5, 0.5, 0.5), vec3(2, 2, 2))));
    EXPECT_FALSE(Intersection::AABBAABB(box, AABB(vec3(1.5, 0.5, 0.5), vec3(2, 2, 2))));
}

TEST(Geometry, AABBFunctions)
{
    AABB box;
    box.makeNegative();
    box.growBox(vec3(1, 2, 3));
    box.growBox(vec3(-1, 0, 5));
    EXPECT_EQ(box.min, vec3(-1, 0, 3));
    EXPECT_EQ(box.max, vec3(1, 2, 5));
    EXPECT_EQ(box.maxDimension(), 0);
    EXPECT_EQ(box.maxSize(), 2);
    EXPECT_EQ(box.getPosition(), vec3(0, 1, 4));
    EXPECT_TRUE(box.contains(vec3(0, 1, 4)));
    EXPECT_FALSE(box.contains(vec3(0, 1, 6)));
    EXPECT_EQ(box.DistanceSquared(vec3(0, 1, 4)), 0);
    EXPECT_EQ(box.DistanceSquared(vec3(3, 1, 4)), 4);
    EXPECT_EQ(box.DistanceSquared(vec3(3, 4, 4)), 8);

    for (int i = 0; i < 8; ++i)
    {
        vec3 c = box.cornerPoint(i);
        EXPECT_TRUE(box.contains(c));
        EXPECT_EQ(box.DistanceSquared(c), 0);
    }

    auto tris = box.toTriangles();
    ASSERT_EQ(tris.size(), 12u);
    float area = 0;
    for (auto& t : tris) area += t.Area();
    EXPECT_NEAR(area, 2 * (2 * 2 + 2 * 2 + 2 * 2), 1e-5);

    box.setPosition(vec3(10, 10, 10));
    EXPECT_EQ(box.getPosition(), vec3(10, 10, 10));
    box.scale(vec3(2, 1, 1));
    EXPECT_EQ(box.Size(), vec3(4, 2, 2));

    AABB inv(vec3(1, 1, 1), vec3(0, 2, 0));
    inv.ensureValidity();
    EXPECT_EQ(inv.min, vec3(0, 1, 0));
    EXPECT_EQ(inv.max, vec3(1, 2, 1));

    auto [center, radius] = inv.BoundingSphere();
    EXPECT_EQ(center, vec3(0.5, 1.5, 0.5));
    EXPECT_NEAR(radius, std::sqrt(3.f) * 0.5f, 1e-6);
}

TEST(Geometry, ClipTriangleAABB)
{
    AABB box(vec3(0, 0, 0), vec3(1, 1, 1));

    // Completely inside: unchanged
    Triangle inside(vec3(0.1, 0.1, 0.5), vec3(0.9, 0.1, 0.5), vec3(0.5, 0.9, 0.5));
    auto p = Clipping::clipTriAABB(inside, box);
    EXPECT_EQ(p.size(), 3u);

    // Completely outside: empty
    Triangle outside(vec3(2, 2, 2), vec3(3, 2, 2), vec3(2, 3, 2));
    EXPECT_TRUE(Clipping::clipTriAABB(outside, box).empty());

    // A large triangle in the plane z = 0.5 covering the box: the result is the square of the box.
    Triangle large(vec3(-10, -10, 0.5), vec3(10, -10, 0.5), vec3(0, 20, 0.5));
    p = Clipping::clipTriAABB(large, box);
    EXPECT_GE(p.size(), 4u);
    AABB bb = Clipping::clipTriAABBtoBox(large, box);
    EXPECT_NEAR((bb.min - vec3(0, 0, 0.5)).norm(), 0, 1e-5);
    EXPECT_NEAR((bb.max - vec3(1, 1, 0.5)).norm(), 0, 1e-5);
    for (auto& v : p)
    {
        EXPECT_GE(v.minCoeff(), -1e-5);
        EXPECT_LE(v.maxCoeff(), 1 + 1e-5);
    }
}

TEST(Geometry, FrustumIntersectSAT)
{
    Frustum f1(mat4::Identity(), radians(60.f), 1, 1, 10);
    // The same frustum moved far to the side does not intersect.
    Frustum f2(translate(vec3(100, 0, 0)), radians(60.f), 1, 1, 10);
    // Moved a little: intersects.
    Frustum f3(translate(vec3(1, 0, 0)), radians(60.f), 1, 1, 10);
    EXPECT_TRUE(f1.intersectSAT(f1));
    EXPECT_FALSE(f1.intersectSAT(f2));
    EXPECT_TRUE(f1.intersectSAT(f3));

    EXPECT_TRUE(f1.intersectSAT(Sphere(vec3(0, 0, -5), 0.5)));
    EXPECT_FALSE(f1.intersectSAT(Sphere(vec3(0, 0, 5), 0.5)));
    EXPECT_FALSE(f1.intersectSAT(Sphere(vec3(20, 0, -5), 0.5)));

    // The bounding sphere contains all corners.
    for (auto& v : f1.vertices)
    {
        EXPECT_LE((v - f1.boundingSphere.pos).norm(), f1.boundingSphere.r + 1e-4);
    }
    EXPECT_EQ(f1.pointInSphereFrustum(vec3(0, 0, -5)), Frustum::INSIDE);
    EXPECT_EQ(f1.pointInSphereFrustum(vec3(0, 0, 50)), Frustum::OUTSIDE);

    EXPECT_EQ(f1.sideOfPlane(Plane(vec3(0, 0, 1), vec3(0, 0, 1))), -1);
    EXPECT_EQ(f1.sideOfPlane(Plane(vec3(0, 0, -20), vec3(0, 0, 1))), 1);
    EXPECT_EQ(f1.sideOfPlane(Plane(vec3(0, 0, -5), vec3(0, 0, 1))), 0);
}

TEST(Geometry, BVHMatchesBruteForce)
{
    Random::setSeed(42);
    std::vector<Triangle> triangles;
    for (int i = 0; i < 300; ++i)
    {
        vec3 c = Random::MatrixUniform<vec3>(-5, 5);
        triangles.emplace_back(c + Random::MatrixUniform<vec3>(-0.5, 0.5), c + Random::MatrixUniform<vec3>(-0.5, 0.5),
                               c + Random::MatrixUniform<vec3>(-0.5, 0.5));
    }

    AccelerationStructure::BruteForce brute(triangles);
    AccelerationStructure::ObjectMedianBVH bvh(triangles, 4);

    int hits = 0;
    for (int i = 0; i < 300; ++i)
    {
        // Every second ray points towards a triangle, so that there are enough hits.
        vec3 origin = Random::MatrixUniform<vec3>(-7, 7);
        vec3 dir    = Random::sphericalRand(1).cast<float>();
        if (i % 2 == 0) dir = (triangles[i].center() - origin).normalized();
        Ray r(dir, origin);
        auto a = brute.getClosest(r);
        auto b = bvh.getClosest(r);
        ASSERT_EQ(a.valid, b.valid) << i;
        if (a.valid)
        {
            hits++;
            EXPECT_NEAR(a.t, b.t, 1e-4) << i;
            EXPECT_EQ(a.triangleIndex, b.triangleIndex) << i;
        }
        EXPECT_EQ(brute.getAll(r).size(), bvh.getAll(r).size()) << i;

        vec3 p    = Random::MatrixUniform<vec3>(-7, 7);
        auto cp   = bvh.ClosestPoint(p);
        float ref = std::numeric_limits<float>::infinity();
        for (auto& t : triangles) ref = std::min(ref, t.Distance(p));
        EXPECT_NEAR(cp.first, ref, 1e-4) << i;
        EXPECT_NEAR(triangles[cp.second].Distance(p), ref, 1e-4) << i;
    }
    EXPECT_GT(hits, 150);

    // An empty structure never hits.
    AccelerationStructure::ObjectMedianBVH empty(std::vector<Triangle>{}, 4);
    EXPECT_FALSE(empty.getClosest(Ray(vec3(1, 0, 0), vec3(0, 0, 0))).valid);
}

TEST(Geometry, BezierAndBspline)
{
    Bezier<vec2> bez({vec2(0, 0), vec2(1, 2), vec2(3, 2), vec2(4, 0)});
    EXPECT_EQ(bez.evaluate(0), vec2(0, 0));
    EXPECT_EQ(bez.evaluate(1), vec2(4, 0));
    // B(0.5) = (P0 + 3 P1 + 3 P2 + P3) / 8
    EXPECT_NEAR((bez.evaluate(0.5) - vec2(2, 1.5)).norm(), 0, 1e-6);

    // Degree elevation and subdivision don't change the curve.
    Bezier<vec2> elevated = bez;
    elevated.degreeElavation();
    EXPECT_EQ(elevated.N, 4);
    auto halves = bez.subdivide(1);
    ASSERT_EQ(halves.size(), 2u);
    for (float t : {0.f, 0.2f, 0.5f, 0.7f, 1.f})
    {
        EXPECT_NEAR((elevated.evaluate(t) - bez.evaluate(t)).norm(), 0, 1e-5) << t;
        EXPECT_NEAR((halves[0].evaluate(t) - bez.evaluate(t * 0.5f)).norm(), 0, 1e-5) << t;
        EXPECT_NEAR((halves[1].evaluate(t) - bez.evaluate(0.5f + t * 0.5f)).norm(), 0, 1e-5) << t;
    }
    EXPECT_EQ(bez.createLine(2).size(), 4u * 3 + 1);

    // A B-spline with end interpolation starts and ends at the control points. A cubic B-spline with 4 control
    // points and clamped knots is the Bezier curve.
    Bspline<vec2, float> spline({vec2(0, 0), vec2(1, 2), vec2(3, 2), vec2(4, 0)}, 3);
    EXPECT_NEAR((spline.getPointOnCurve(0) - vec2(0, 0)).norm(), 0, 1e-5);
    EXPECT_NEAR((spline.getPointOnCurve(1) - vec2(4, 0)).norm(), 0, 1e-5);
    for (float t : {0.1f, 0.5f, 0.8f})
    {
        EXPECT_NEAR((spline.getPointOnCurve(t) - bez.evaluate(t)).norm(), 0, 1e-5) << t;
    }
}

TEST(Geometry, KDTreeMatchesBruteForce)
{
    Random::setSeed(3);
    std::vector<vec3> points;
    for (int i = 0; i < 500; ++i) points.push_back(Random::MatrixUniform<vec3>(-1, 1));
    KDTree<3, vec3> tree(points);

    for (int i = 0; i < 50; ++i)
    {
        vec3 q = Random::MatrixUniform<vec3>(-1.2, 1.2);
        std::vector<std::pair<float, int>> dist;
        for (int j = 0; j < (int)points.size(); ++j) dist.push_back({(points[j] - q).squaredNorm(), j});
        std::sort(dist.begin(), dist.end());

        EXPECT_EQ(tree.NearestNeighborSearch(q), dist[0].second);

        auto knn = tree.KNearestNeighborSearch(q, 5);
        ASSERT_EQ(knn.size(), 5u);
        for (int k = 0; k < 5; ++k) EXPECT_EQ(knn[k], dist[k].second);

        float r     = 0.3;
        auto radius = tree.RadiusSearch(q, r);
        std::vector<int> expected;
        for (auto& d : dist)
            if (d.first < r * r) expected.push_back(d.second);
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(radius, expected);
    }
}

// Regression test: AABB::transform only transformed min and max, which is wrong for rotations and negative scales.
TEST(Geometry, AABBTransform)
{
    AABB box(vec3(-1, -2, -3), vec3(1, 2, 3));

    // 90 degree rotation around z: x extent and y extent swap.
    AABB r = box;
    r.transform(translate(vec3(10, 0, 0)) * Saiga::rotate(radians(90.f), vec3(0, 0, 1)));
    EXPECT_NEAR((r.min - vec3(8, -1, -3)).norm(), 0, 1e-5) << r.min.transpose();
    EXPECT_NEAR((r.max - vec3(12, 1, 3)).norm(), 0, 1e-5) << r.max.transpose();

    // Negative scale must keep min <= max.
    AABB s(vec3(1, 1, 1), vec3(2, 3, 4));
    s.transform(scale(vec3(-1, 2, 1)));
    EXPECT_EQ(s.min, vec3(-2, 2, 1));
    EXPECT_EQ(s.max, vec3(-1, 6, 4));

    // A general rotation: the result must be the bounding box of the transformed corners.
    mat4 T = translate(vec3(1, 2, 3)) * Saiga::rotate(radians(30.f), vec3(1, 2, 3).normalized());
    AABB expected;
    expected.makeNegative();
    for (int i = 0; i < 8; ++i)
    {
        expected.growBox(make_vec3(T * make_vec4(box.cornerPoint(i), 1)));
    }
    AABB t = box;
    t.transform(T);
    EXPECT_NEAR((t.min - expected.min).norm(), 0, 1e-5);
    EXPECT_NEAR((t.max - expected.max).norm(), 0, 1e-5);
}

// Regression test: Mesh::normalizeScale divided by (largest coordinate - smallest coordinate) over all axes instead
// of by the largest extent of the bounding box.
TEST(Geometry, MeshNormalizeScale)
{
    TriangleMesh<Vertex, uint32_t> mesh;
    mesh.addVertex(Vertex(vec3(10, 0, 0)));
    mesh.addVertex(Vertex(vec3(11, 1, 0)));
    mesh.addVertex(Vertex(vec3(10, 0, 2)));
    mesh.addFace(0, 1, 2);
    mesh.normalizeScale();
    EXPECT_NEAR(mesh.aabb().maxSize(), 1, 1e-6);

    mesh.normalizeScale(0.25);
    EXPECT_NEAR(mesh.aabb().maxSize(), 1 / 1.5, 1e-6);
}
