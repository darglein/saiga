#pragma once

#include "libhello/opengl/texture/texture.h"
#include "libhello/opengl/texture/cube_texture.h"
#include "libhello/opengl/indexedVertexBuffer.h"
#include "libhello/opengl/basic_shaders.h"

#include "libhello/world/heightmap.h"
#include "libhello/world/terrainmesh.h"
#include "libhello/world/clipmap.h"
#include "libhello/camera/camera.h"

#include "libhello/opengl/texture/freeimage.h"

#include "glm/gtc/random.hpp"

class Terrain{
public:
    int layers;

    TerrainShader* shader;
    TerrainShader* deferredshader;
    TerrainShader* depthshader;

    Texture* texture1, *texture2;

    Heightmap heightmap;
    mat4 model;

    vec3 viewPos;

    std::vector<Clipmap> clipmaps;
//    Clipmap clipmaps[8];
    vec2 baseScale = vec2(1,1);



    Terrain(int layers, int w, int h , float heightScale);

    bool loadHeightmap();
    void createHeightmap();

    void createMesh();

    void setDistance(float d);

    void render(Camera* cam);
    void renderDepth(Camera* cam);

    void update(const vec3& p);
private:

    void renderintern(Camera* cam);

    void render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color, vec4 scale,vec4 fineOrigin);

};
