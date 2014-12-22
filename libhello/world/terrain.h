#pragma once

#include "libhello/opengl/texture/texture.h"
#include "libhello/opengl/texture/cube_texture.h"
#include "libhello/opengl/indexedVertexBuffer.h"
#include "libhello/opengl/basic_shaders.h"

#include "libhello/world/heightmap.h"
#include "libhello/world/terrainmesh.h"
#include "libhello/world/clipmap.h"
#include "libhello/camera/camera.h"

#include "glm/gtc/random.hpp"

class Terrain{
public:
    const int levels = 8;

    TerrainShader* shader;
    TerrainShader* deferredshader;
    TerrainShader* depthshader;
//    std::vector<Texture*> texture;

    Heightmap heightmap;
    mat4 model;

    vec3 viewPos;

    Clipmap clipmaps[8];
    vec2 baseScale = vec2(1,1);

    Terrain();

    void createMesh(unsigned int w, unsigned int h);

    void setDistance(float d);

    void render(Camera* cam);
    void renderDepth(Camera* cam);

    void update(const vec3& p);
private:

    void renderintern(Camera* cam);

    void render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color, vec4 scale,vec4 fineOrigin);

};
