/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/common/GameMath.h"

#include <cstdint>
#include <vector>

namespace Saiga
{
namespace Game
{
// A vertex of the flat colored low-poly meshes (see MeshRenderer)
struct MeshVertex
{
    float position[3];
    float normal[3];
    float color[3];
};

/**
 * Triangle mesh with one normal per face (flat shading: the vertices of a face are not shared) and a color per
 * vertex. Built on the CPU by the procedural builders below, drawn by MeshRenderer.
 */
struct MeshData
{
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;

    void addTriangle(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& color);
    // a, b, c, d in order around the quad
    void addQuad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, const Vec3& color);
    // Appends other transformed by m (positions and normals)
    void append(const MeshData& other, const Mat4& m = Mat4::Identity());
};

// Procedural low-poly meshes. Forward is -z, up is +y.
namespace MeshBuilder
{
MeshData box(const Vec3& halfExtents, const Vec3& color, const Vec3& center = Vec3(0, 0, 0));
// Along the y axis from y = -halfLength to +halfLength
MeshData cylinder(double radius, double halfLength, int segments, const Vec3& color);
// Base at y = 0, apex at y = height
MeshData cone(double radius, double height, int segments, const Vec3& color);
// A low-poly rock: an icosahedron of radius r (its centre at the origin) with a fixed jitter of its vertices (the same
// mesh every run), flat faces of the color
MeshData rock(double radius, const Vec3& color);
// A small propeller airplane, nose along -z, about 9 m wingspan
MeshData airplane();
// A car body without wheels: 1.8 m wide, 4.2 m long, the origin at the center of mass
MeshData carBody(const Vec3& color);
// A wheel: a cylinder along x with radius r and a lighter hub
MeshData wheel(double radius, double width);
// A traffic cone marker (0.8 m)
MeshData trafficCone();
// A flag pole with a flag (3 m), base at the origin
MeshData flag(const Vec3& color);
}  // namespace MeshBuilder

}  // namespace Game
}  // namespace Saiga
