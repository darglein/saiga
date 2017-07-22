/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#define _USE_MATH_DEFINES
#include "saiga/geometry/triangle_mesh.h"
#include "saiga/geometry/sphere.h"
#include "saiga/geometry/plane.h"
#include "saiga/geometry/cone.h"

namespace Saiga {

class SAIGA_GLOBAL TriangleMeshGenerator
{
    typedef TriangleMesh<VertexNT,GLuint>::Face Face;

public:

    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createMesh(const Sphere &sphere, int rings, int sectors);
    //TODO: uv mapping
    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createMesh(const Sphere &sphere, int resolution);

     static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createCylinderMesh(float radius, float height, int sectors);

    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createMesh(const Plane &plane);

    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createFullScreenQuadMesh();
    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createQuadMesh();

    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createMesh(const Cone &cone, int sectors);

    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createMesh(const AABB &box);
    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createSkyboxMesh(const AABB &box);


    static std::shared_ptr<TriangleMesh<VertexNT,GLuint>> createGridMesh(unsigned int w, unsigned int h);

    template<typename vertex_t, typename index_t>
    static std::shared_ptr<TriangleMesh<vertex_t,index_t>> createGridMesh2(int w, int h);



};



template<typename vertex_t, typename index_t>
std::shared_ptr<TriangleMesh<vertex_t,index_t>>  TriangleMeshGenerator::createGridMesh2(int w, int h){
    TriangleMesh<vertex_t,index_t>* mesh = new TriangleMesh<vertex_t,index_t>();

    //creating uniform grid with w*h vertices
    //the resulting mesh will fill the quad (-1,0,-1) - (1,0,1)
    float dw = (2.0/w);
    float dh = (2.0/h);
    for(unsigned int y=0;y<h;y++){
        for(unsigned int x=0;x<w;x++){
            float fx = (float)x*dw-1.0f;
            float fy = (float)y*dh-1.0f;
            vertex_t v;
            v.position = vec3(fx,0.0f,fy);
            mesh->addVertex(v);
        }
    }


    for(unsigned int y=0;y<h-1;y++){
        for(unsigned int x=0;x<w-1;x++){
            GLuint quad[] = {y*w+x,(y+1)*w+x,(y+1)*w+x+1,y*w+x+1};
            mesh->addQuad(quad);
        }
    }

    return std::shared_ptr<TriangleMesh<vertex_t,index_t>>(mesh);
}

}
