#pragma once

#include "libhello/opengl/texture/texture.h"
#include "libhello/opengl/texture/cube_texture.h"
#include "libhello/opengl/indexedVertexBuffer.h"
#include "libhello/opengl/basic_shaders.h"
#include "libhello/util/perlinnoise.h"
#include "libhello/geometry/triangle_mesh_generator.h"


class Heightmap{
private:
    typedef TriangleMesh<Vertex,GLuint> mesh_t;
public:
    int n = 63;
     int m = (n+1)/4;
    Image heightmap;
    Image normalmap;

    Heightmap(int w, int h);

    void createNoiseHeightmap();
    void createTestHeightmap();
    Texture* createTexture();

    std::shared_ptr<mesh_t> createMesh();
    std::shared_ptr<mesh_t> createMesh2();
     std::shared_ptr<Heightmap::mesh_t> createMeshFixUpV();
      std::shared_ptr<Heightmap::mesh_t> createMeshFixUpH();
      std::shared_ptr<Heightmap::mesh_t> createMeshTrim();
       std::shared_ptr<Heightmap::mesh_t> createMeshTrimi();
       std::shared_ptr<Heightmap::mesh_t> createMeshCenter();

      std::shared_ptr<mesh_t> createGridMesh(unsigned int w, unsigned int h, vec2 d, vec2 o);
};
