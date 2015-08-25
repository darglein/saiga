#pragma once

#include "saiga/opengl/texture/texture.h"
#include "saiga/opengl/texture/cube_texture.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"

#include "saiga/world/heightmap.h"
#include "saiga/world/terrainmesh.h"
#include "saiga/world/clipmap.h"
#include "saiga/camera/camera.h"


#include "glm/gtc/random.hpp"

class SAIGA_GLOBAL Terrain{
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
