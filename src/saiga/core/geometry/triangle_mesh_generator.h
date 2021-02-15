/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "cone.h"
#include "plane.h"
#include "sphere.h"

#include <memory>

#include "triangle_mesh.h"
#include "triangle_mesh_generator2.h"
namespace Saiga
{
class SAIGA_CORE_API TriangleMeshGenerator
{
    typedef TriangleMesh<VertexNT, uint32_t>::Face Face;

   public:
    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createMesh(const Sphere& sphere, int rings, int sectors);
    // TODO: uv mapping
    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createMesh(const Sphere& sphere, int resolution);

    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createCylinderMesh(float radius, float height,
                                                                                int sectors);

    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createMesh(const Plane& plane);

    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createFullScreenQuadMesh();

    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createMesh(const Cone& cone, int sectors);

    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createMesh(const AABB& box);
    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createSkyboxMesh(const AABB& box);
};


}  // namespace Saiga
