/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Mesh data of the tech demo games and the procedural low-poly mesh builders.

#include "games/common/Mesh.h"

namespace Saiga
{
namespace Game
{
namespace
{
MeshVertex vertex(const Vec3& p, const Vec3& n, const Vec3& c)
{
    MeshVertex v;
    for (int i = 0; i < 3; ++i)
    {
        v.position[i] = float(p(i));
        v.normal[i]   = float(n(i));
        v.color[i]    = float(c(i));
    }
    return v;
}
}  // namespace

void MeshData::addTriangle(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& color)
{
    Vec3 n   = (b - a).cross(c - a);
    double l = n.norm();
    n        = l > 1e-12 ? Vec3(n / l) : Vec3(0, 1, 0);
    auto i   = uint32_t(vertices.size());
    vertices.push_back(vertex(a, n, color));
    vertices.push_back(vertex(b, n, color));
    vertices.push_back(vertex(c, n, color));
    indices.insert(indices.end(), {i, i + 1, i + 2});
}

void MeshData::addQuad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, const Vec3& color)
{
    addTriangle(a, b, c, color);
    addTriangle(a, c, d, color);
}

void MeshData::append(const MeshData& other, const Mat4& m)
{
    // Normals with the inverse transpose of the upper 3x3 (non uniform scale)
    Mat3 A;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) A(i, j) = m(i, j);
    Mat3 N = A.inverse().transpose();

    auto base = uint32_t(vertices.size());
    for (auto v : other.vertices)
    {
        Vec3 p = transformPoint(m, Vec3(v.position[0], v.position[1], v.position[2]));
        Vec3 n = N * Vec3(v.normal[0], v.normal[1], v.normal[2]);
        n.normalize();
        for (int i = 0; i < 3; ++i)
        {
            v.position[i] = float(p(i));
            v.normal[i]   = float(n(i));
        }
        vertices.push_back(v);
    }
    for (auto i : other.indices) indices.push_back(base + i);
}

namespace MeshBuilder
{
MeshData box(const Vec3& h, const Vec3& color, const Vec3& c)
{
    MeshData m;
    auto P = [&](double x, double y, double z) { return Vec3(c(0) + x * h(0), c(1) + y * h(1), c(2) + z * h(2)); };
    // Counter clockwise seen from outside
    m.addQuad(P(-1, -1, 1), P(1, -1, 1), P(1, 1, 1), P(-1, 1, 1), color);      // +z
    m.addQuad(P(1, -1, -1), P(-1, -1, -1), P(-1, 1, -1), P(1, 1, -1), color);  // -z
    m.addQuad(P(1, -1, 1), P(1, -1, -1), P(1, 1, -1), P(1, 1, 1), color);      // +x
    m.addQuad(P(-1, -1, -1), P(-1, -1, 1), P(-1, 1, 1), P(-1, 1, -1), color);  // -x
    m.addQuad(P(-1, 1, 1), P(1, 1, 1), P(1, 1, -1), P(-1, 1, -1), color);      // +y
    m.addQuad(P(-1, -1, -1), P(1, -1, -1), P(1, -1, 1), P(-1, -1, 1), color);  // -y
    return m;
}

MeshData cylinder(double r, double h, int segments, const Vec3& color)
{
    MeshData m;
    for (int i = 0; i < segments; ++i)
    {
        double a0 = 2 * kPi * i / segments, a1 = 2 * kPi * (i + 1) / segments;
        Vec3 p0(r * std::cos(a0), 0, r * std::sin(a0)), p1(r * std::cos(a1), 0, r * std::sin(a1));
        Vec3 up(0, h, 0);
        m.addQuad(p0 - up, p0 + up, p1 + up, p1 - up, color);
        m.addTriangle(up, p1 + up, p0 + up, color);
        m.addTriangle(-up, p0 - up, p1 - up, color);
    }
    return m;
}

MeshData cone(double r, double height, int segments, const Vec3& color)
{
    MeshData m;
    Vec3 apex(0, height, 0);
    for (int i = 0; i < segments; ++i)
    {
        double a0 = 2 * kPi * i / segments, a1 = 2 * kPi * (i + 1) / segments;
        Vec3 p0(r * std::cos(a0), 0, r * std::sin(a0)), p1(r * std::cos(a1), 0, r * std::sin(a1));
        m.addTriangle(p0, apex, p1, color);
        m.addTriangle(Vec3(0, 0, 0), p0, p1, color);
    }
    return m;
}

MeshData airplane()
{
    Vec3 body(0.85, 0.85, 0.82), accent(0.75, 0.2, 0.15), glass(0.25, 0.3, 0.35), dark(0.2, 0.2, 0.22);
    MeshData m;
    // Fuselage, tapering to the tail
    m.append(box(Vec3(0.55, 0.6, 2.2), body, Vec3(0, 0, -0.6)));
    m.append(box(Vec3(0.35, 0.4, 1.6), body, Vec3(0, 0.1, 3.0)));
    m.append(box(Vec3(0.4, 0.3, 0.7), glass, Vec3(0, 0.75, -0.9)));
    // Engine cowling and propeller
    m.append(box(Vec3(0.5, 0.5, 0.3), accent, Vec3(0, 0, -3.05)));
    m.append(box(Vec3(1.2, 0.12, 0.05), dark, Vec3(0, 0, -3.4)));
    // Wings with colored tips
    m.append(box(Vec3(4.5, 0.08, 0.8), body, Vec3(0, 0.1, -0.7)));
    m.append(box(Vec3(0.5, 0.09, 0.8), accent, Vec3(4.3, 0.1, -0.7)));
    m.append(box(Vec3(0.5, 0.09, 0.8), accent, Vec3(-4.3, 0.1, -0.7)));
    // Tail: horizontal stabilizer and fin
    m.append(box(Vec3(1.6, 0.06, 0.45), body, Vec3(0, 0.2, 4.2)));
    m.append(box(Vec3(0.06, 0.8, 0.5), accent, Vec3(0, 0.9, 4.2)));
    // Landing gear
    m.append(box(Vec3(0.05, 0.4, 0.05), dark, Vec3(0.7, -0.9, -1.0)));
    m.append(box(Vec3(0.05, 0.4, 0.05), dark, Vec3(-0.7, -0.9, -1.0)));
    m.append(box(Vec3(0.12, 0.2, 0.2), dark, Vec3(0.7, -1.3, -1.0)));
    m.append(box(Vec3(0.12, 0.2, 0.2), dark, Vec3(-0.7, -1.3, -1.0)));
    return m;
}

MeshData carBody(const Vec3& color)
{
    Vec3 glass(0.2, 0.24, 0.3), dark(0.15, 0.15, 0.17), light(0.95, 0.9, 0.6);
    MeshData m;
    m.append(box(Vec3(0.9, 0.3, 2.1), color, Vec3(0, -0.05, 0)));
    m.append(box(Vec3(0.78, 0.28, 1.0), glass, Vec3(0, 0.5, 0.2)));
    m.append(box(Vec3(0.8, 0.04, 1.0), color, Vec3(0, 0.8, 0.2)));
    // Spoiler, bumpers, lights
    m.append(box(Vec3(0.85, 0.04, 0.2), dark, Vec3(0, 0.55, 1.95)));
    m.append(box(Vec3(0.92, 0.1, 0.08), dark, Vec3(0, -0.25, -2.12)));
    m.append(box(Vec3(0.92, 0.1, 0.08), dark, Vec3(0, -0.25, 2.12)));
    m.append(box(Vec3(0.18, 0.06, 0.03), light, Vec3(0.6, 0.05, -2.12)));
    m.append(box(Vec3(0.18, 0.06, 0.03), light, Vec3(-0.6, 0.05, -2.12)));
    return m;
}

MeshData wheel(double radius, double width)
{
    MeshData m;
    // The cylinder is along y: rotate it to x
    Mat4 toX = transformMatrix(Vec3(0, 0, 0), axisAngle(Vec3(0, 0, 1), kPi / 2));
    m.append(cylinder(radius, width / 2, 10, Vec3(0.12, 0.12, 0.13)), toX);
    m.append(cylinder(radius * 0.55, width / 2 + 0.01, 6, Vec3(0.7, 0.7, 0.72)), toX);
    return m;
}

MeshData trafficCone()
{
    MeshData m = cone(0.25, 0.8, 8, Vec3(0.95, 0.45, 0.1));
    m.append(box(Vec3(0.3, 0.03, 0.3), Vec3(0.2, 0.2, 0.2), Vec3(0, 0.03, 0)));
    return m;
}

MeshData flag(const Vec3& color)
{
    MeshData m = box(Vec3(0.05, 1.5, 0.05), Vec3(0.8, 0.8, 0.8), Vec3(0, 1.5, 0));
    m.append(box(Vec3(0.6, 0.35, 0.02), color, Vec3(0.65, 2.6, 0)));
    return m;
}
}  // namespace MeshBuilder

}  // namespace Game
}  // namespace Saiga
