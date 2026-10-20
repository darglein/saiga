/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/model/UnifiedMesh.h"
#include "saiga/core/model/UnifiedModel.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/model/model_loader_off.h"
#include "saiga/core/model/model_loader_ply.h"

#include "gtest/gtest.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace Saiga;

// Regression test: operator<< of UnifiedMaterial and UnifiedModel wrote to std::cout instead of the given stream.
TEST(UnifiedModel, StreamOperatorsWriteToStream)
{
    UnifiedMaterial mat("my_material");
    std::stringstream strm;
    strm << mat;
    EXPECT_NE(strm.str().find("my_material"), std::string::npos);

    UnifiedModel model;
    model.name = "my_model";
    model.materials.push_back(mat);
    std::stringstream strm2;
    strm2 << model;
    EXPECT_NE(strm2.str().find("my_model"), std::string::npos);
    EXPECT_NE(strm2.str().find("Materials"), std::string::npos);
    EXPECT_NE(strm2.str().find("my_material"), std::string::npos);
}

// Regression test: EraseVertices (and therefore RemoveDoubles) updated the triangle indices but not the line indices.
TEST(UnifiedMesh, EraseVerticesUpdatesLines)
{
    UnifiedMesh mesh;
    mesh.position  = {vec3(0, 0, 0), vec3(1, 0, 0), vec3(2, 0, 0), vec3(3, 0, 0)};
    mesh.lines     = {ivec2(0, 1), ivec2(2, 3), ivec2(1, 3)};
    mesh.triangles = {ivec3(1, 2, 3), ivec3(0, 2, 3)};

    std::vector<int> erase = {0};
    mesh.EraseVertices(erase);
    ASSERT_EQ(mesh.NumVertices(), 3);
    EXPECT_EQ(mesh.triangles, std::vector<ivec3>({ivec3(0, 1, 2)}));
    // The line that used vertex 0 is removed, the others are shifted.
    EXPECT_EQ(mesh.lines, std::vector<ivec2>({ivec2(1, 2), ivec2(0, 2)}));

    UnifiedMesh mesh2;
    mesh2.position = {vec3(0, 0, 0), vec3(1, 0, 0), vec3(1, 0, 0), vec3(2, 0, 0)};
    mesh2.lines    = {ivec2(0, 1), ivec2(2, 3)};
    mesh2.RemoveDoubles(0.01);
    ASSERT_EQ(mesh2.NumVertices(), 3);
    EXPECT_EQ(mesh2.lines, std::vector<ivec2>({ivec2(0, 1), ivec2(1, 2)}));
}

// Regression test: ReorderMorton64 added the bounding box minimum instead of subtracting it, so the Morton codes of
// meshes with a negative bounding box were computed from negative (wrapped) coordinates.
TEST(UnifiedMesh, ReorderMorton64)
{
    UnifiedMesh mesh;
    std::vector<int> order = {7, 2, 9, 0, 4, 1, 8, 3, 6, 5};
    for (int i : order)
    {
        float t = -10.f + i;
        mesh.position.push_back(vec3(t, 2 * t, 0.5f * t));
    }
    mesh.ReorderMorton64();
    ASSERT_EQ(mesh.NumVertices(), 10);
    // All coordinates grow with t, so the Morton order is the order of t.
    for (int i = 0; i + 1 < mesh.NumVertices(); ++i)
    {
        EXPECT_LT(mesh.position[i].x(), mesh.position[i + 1].x()) << i;
    }
}

// Regression test: SmoothVertexColors divided by zero for vertices that are not part of a triangle.
TEST(UnifiedMesh, SmoothVertexColorsIsolatedVertex)
{
    UnifiedMesh mesh;
    mesh.position  = {vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), vec3(5, 5, 5)};
    mesh.color     = {vec4(1, 0, 0, 1), vec4(0, 1, 0, 1), vec4(0, 0, 1, 1), vec4(1, 1, 1, 1)};
    mesh.triangles = {ivec3(0, 1, 2)};
    mesh.SmoothVertexColors(1, 1);
    for (auto& c : mesh.color) EXPECT_TRUE(c.allFinite());
    EXPECT_NEAR((mesh.color[0] - vec4(1.f / 3, 1.f / 3, 1.f / 3, 1)).norm(), 0, 1e-6);
    EXPECT_NEAR((mesh.color[3] - vec4(1, 1, 1, 1)).norm(), 0, 1e-6);
}

// Regression test: transform() multiplied the normals with the model matrix, so a scaling changed their length and a
// non-uniform scaling tilted them.
TEST(UnifiedMesh, TransformNormals)
{
    UnifiedMesh mesh;
    mesh.position  = {vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0)};
    mesh.triangles = {ivec3(0, 1, 2)};
    mesh.normal    = {vec3(1, 1, 0).normalized(), vec3(0, 0, 1), vec3(0, 0, 0)};
    mesh.transform(translate(vec3(1, 2, 3)) * scale(vec3(2, 1, 1)));

    EXPECT_NEAR((mesh.position[1] - vec3(3, 2, 3)).norm(), 0, 1e-6);
    // The plane x + y = 0 becomes x / 2 + y = 0, which has the normal (1, 2, 0) / sqrt(5).
    EXPECT_NEAR((mesh.normal[0] - vec3(1, 2, 0).normalized()).norm(), 0, 1e-6);
    EXPECT_NEAR((mesh.normal[1] - vec3(0, 0, 1)).norm(), 0, 1e-6);
    // Zero normals stay zero.
    EXPECT_EQ(mesh.normal[2], vec3(0, 0, 0));
}

static void ExpectClosedSphere(const UnifiedMesh& mesh, const Sphere& s)
{
    ASSERT_GT(mesh.NumVertices(), 0);
    ASSERT_GT(mesh.NumFaces(), 0);
    for (int i = 0; i < mesh.NumVertices(); ++i)
    {
        EXPECT_NEAR((mesh.position[i] - s.pos).norm(), s.r, 1e-4 * s.r) << i;
        EXPECT_NEAR(mesh.normal[i].norm(), 1, 1e-4) << i;
        EXPECT_NEAR((mesh.normal[i] - (mesh.position[i] - s.pos).normalized()).norm(), 0, 1e-4) << i;
    }
    // The triangles are ordered counter clockwise when seen from the outside.
    for (auto t : mesh.triangles)
    {
        Triangle tri(mesh.position[t(0)], mesh.position[t(1)], mesh.position[t(2)]);
        EXPECT_GT(tri.normal().dot(tri.center() - s.pos), 0);
    }
    // The volume of the closed mesh (divergence theorem) is close to the sphere volume.
    double volume = 0;
    for (auto t : mesh.triangles)
    {
        Vec3 a = (mesh.position[t(0)] - s.pos).cast<double>();
        Vec3 b = (mesh.position[t(1)] - s.pos).cast<double>();
        Vec3 c = (mesh.position[t(2)] - s.pos).cast<double>();
        volume += a.dot(b.cross(c)) / 6.0;
    }
    double sphere_volume = 4.0 / 3.0 * pi<double>() * s.r * s.r * s.r;
    EXPECT_NEAR(volume / sphere_volume, 1, 0.05);
}

// Regression test: the normals of IcoSphereMesh had the length of the radius, because transform() scaled them.
TEST(UnifiedMesh, IcoSphere)
{
    Sphere s(vec3(-1, 0.5, 2), 0.5);
    ExpectClosedSphere(IcoSphereMesh(s, 2), s);
}

// Regression test: UVSphereMesh placed all vertices on a cone (the y coordinate used sin(-2 pi + ...) instead of
// -cos(...)) and then multiplied the model with the scalar 1 / sectors instead of the sphere radius.
TEST(UnifiedMesh, UVSphere)
{
    Sphere s(vec3(1, 2, 3), 2);
    ExpectClosedSphere(UVSphereMesh(s, 20, 30), s);
}
