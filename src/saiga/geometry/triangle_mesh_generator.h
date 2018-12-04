/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/geometry/cone.h"
#include "saiga/geometry/plane.h"
#include "saiga/geometry/sphere.h"
#include "saiga/geometry/triangle_mesh.h"

namespace Saiga
{
class SAIGA_GLOBAL TriangleMeshGenerator
{
    typedef TriangleMesh<VertexNT, uint32_t>::Face Face;

   public:
    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createMesh(const Sphere& sphere, int rings, int sectors);
    // TODO: uv mapping
    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createMesh(const Sphere& sphere, int resolution);

    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createCylinderMesh(float radius, float height,
                                                                                int sectors);

    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createMesh(const Plane& plane);
    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createTesselatedPlane(int verticesX, int verticesY);

    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createFullScreenQuadMesh();
    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createQuadMesh();

    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createMesh(const Cone& cone, int sectors);

    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createMesh(const AABB& box);
    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createSkyboxMesh(const AABB& box);


    static std::shared_ptr<TriangleMesh<VertexNT, uint32_t>> createGridMesh(unsigned int w, unsigned int h);

    template <typename vertex_t, typename index_t>
    static std::shared_ptr<TriangleMesh<vertex_t, index_t>> createGridMesh2(int w, int h);


    // a circle at the origin parallel to the x-z plane
    template <typename vertex_t, typename index_t>
    static std::shared_ptr<TriangleMesh<vertex_t, index_t>> createCircleMesh(int segments, float radius);
};



template <typename vertex_t, typename index_t>
std::shared_ptr<TriangleMesh<vertex_t, index_t>> TriangleMeshGenerator::createGridMesh2(int w, int h)
{
    //    TriangleMesh<vertex_t,index_t>* mesh = new TriangleMesh<vertex_t,index_t>();
    auto mesh = std::make_shared<TriangleMesh<vertex_t, index_t>>();

    // creating uniform grid with w*h vertices
    // the resulting mesh will fill the quad (-1,0,-1) - (1,0,1)
    float dw = (2.0 / w);
    float dh = (2.0 / h);
    for (unsigned int y = 0; y < h; y++)
    {
        for (unsigned int x = 0; x < w; x++)
        {
            float fx = (float)x * dw - 1.0f;
            float fy = (float)y * dh - 1.0f;
            vertex_t v;
            v.position = vec3(fx, 0.0f, fy);
            mesh->addVertex(v);
        }
    }


    for (unsigned int y = 0; y < h - 1; y++)
    {
        for (unsigned int x = 0; x < w - 1; x++)
        {
            uint32_t quad[] = {y * w + x, (y + 1) * w + x, (y + 1) * w + x + 1, y * w + x + 1};
            mesh->addQuad(quad);
        }
    }

    //    return std::shared_ptr<TriangleMesh<vertex_t,index_t>>(mesh);
    return mesh;
}


template <typename vertex_t, typename index_t>
std::shared_ptr<TriangleMesh<vertex_t, index_t>> TriangleMeshGenerator::createCircleMesh(int segments, float radius)
{
    auto mesh = std::make_shared<TriangleMesh<vertex_t, index_t>>();


    float R = 1. / (float)(segments);
    float r = radius;

    //    vertex_t v;
    //    v.position = vec3(0,0,0);
    //    v.normal = vec3(0,1,0);
    //    mesh->vertices.push_back(v);
    mesh->vertices.push_back(vertex_t(vec3(0, 0, 0), vec3(0, 1, 0)));

    for (int s = 0; s < segments; s++)
    {
        float x = r * sin((float)s * R * glm::two_pi<float>());
        float y = r * cos((float)s * R * glm::two_pi<float>());
        //        v.position = vec3(x,0,y);
        //        v.normal = vec3(0,1,0);
        //        mesh->vertices.push_back(v);
        mesh->vertices.push_back(vertex_t(vec3(x, 0, y), vec3(0, 1, 0)));
    }

    // create a triangle from each 2 neighbouring vertices to the center
    for (int s = 0; s < segments; s++)
    {
        Face face;
        face.v3 = 0;
        face.v2 = ((s + 1) % segments) + 1;
        face.v1 = s + 1;
        mesh->faces.push_back(face);
    }

    return mesh;
}

}  // namespace Saiga
