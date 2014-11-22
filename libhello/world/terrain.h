#pragma once

#include "libhello/opengl/texture/texture.h"
#include "libhello/opengl/texture/cube_texture.h"
#include "libhello/opengl/indexedVertexBuffer.h"
#include "libhello/opengl/basic_shaders.h"


class TerrainShader : public MVPTextureShader{
public:
    TerrainShader(const string &multi_file) : MVPTextureShader(multi_file){}
    virtual void checkUniforms();
};

class Terrain{
public:
    IndexedVertexBuffer<Vertex,GLuint> mesh;
    TerrainShader* shader;
    Texture* texture;
    cube_Texture* cube_texture;
    mat4 model;

    Terrain();

    void createMesh();

    void setPosition(const vec3& p);
    void setDistance(float d);
    void render(const mat4& view, const mat4 &proj);
};
