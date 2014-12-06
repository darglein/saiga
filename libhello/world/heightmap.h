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

     int layers,w,h;

     float* heights;

     float heightScale = 20.0f;
     float minH = 125725;
     float maxH = -0125725;
    std::vector<Image> heightmap;
    std::vector<Image> normalmap;

    std::vector<Texture*> texheightmap;
    std::vector<Texture*> texnormalmap;


    Heightmap(int layers, int w, int h);

    void createTextures();
    void createHeightmaps();
private:

    void createInitialHeightmap();
    void createRemainingLayers();
    void createNormalmap();

    void normalizeHeightMap();

    float getHeight(int x, int y);
    float getHeight(int layer, int x, int y);
    float getHeightScaled(int x, int y);
    float getHeightScaled(int layer, int x, int y);
    void setHeight(int x, int y, float v);


};
