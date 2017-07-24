/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/texture/texture.h"
#include "saiga/opengl/texture/cube_texture.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/util/perlinnoise.h"
#include "saiga/geometry/triangle_mesh_generator.h"

namespace Saiga {

class SAIGA_GLOBAL TerrainMesh{
private:
    typedef TriangleMesh<Vertex,GLuint> mesh_t;
public:
    int n = 63;
    int m = (n+1)/4;


    TerrainMesh();



    std::shared_ptr<mesh_t> createMesh();
    std::shared_ptr<mesh_t> createMesh2();
    std::shared_ptr<TerrainMesh::mesh_t> createMeshFixUpV();
    std::shared_ptr<TerrainMesh::mesh_t> createMeshFixUpH();

    std::shared_ptr<TerrainMesh::mesh_t> createMeshTrimSW();
    std::shared_ptr<TerrainMesh::mesh_t> createMeshTrimSE();

    std::shared_ptr<TerrainMesh::mesh_t> createMeshTrimNW();
    std::shared_ptr<TerrainMesh::mesh_t> createMeshTrimNE();

    std::shared_ptr<TerrainMesh::mesh_t> createMeshCenter();

    std::shared_ptr<TerrainMesh::mesh_t> createMeshDegenerated();

    std::shared_ptr<mesh_t> createGridMesh(unsigned int w, unsigned int h, vec2 d, vec2 o);
};

}
