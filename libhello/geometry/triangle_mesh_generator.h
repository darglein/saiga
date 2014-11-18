#pragma once

#include "libhello/geometry/triangle_mesh.h"
#include "libhello/geometry/sphere.h"
#include "libhello/geometry/plane.h"
#include "libhello/geometry/cone.h"

#include <memory>

class TriangleMeshGenerator
{
    typedef TriangleMesh<VertexNT,GLuint>::Face Face;

public:

    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createMesh(const Sphere &sphere, int rings, int sectors);
    //TODO: uv mapping
    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createMesh(const Sphere &sphere, int resolution);

    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createMesh(const Plane &plane);

    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createFullScreenQuadMesh();
    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createQuadMesh();

    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createMesh(const Cone &cone, int sectors);

    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createMesh(const aabb &box);
    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createSkyboxMesh(const aabb &box);
};
