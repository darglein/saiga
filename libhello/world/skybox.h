#pragma once

#include "libhello/opengl/texture.h"
#include "libhello/opengl/cube_texture.h"
#include "libhello/opengl/indexedVertexBuffer.h"
#include "libhello/opengl/basic_shaders.h"




class Skybox{
public:
    IndexedVertexBuffer<VertexNT,GLuint> mesh;
    MVPTextureShader* shader;
    Texture* texture;
    cube_Texture* cube_texture;
    mat4 model;

    Skybox();

    void setPosition(const vec3& p);
    void setDistance(float d);
    void bindUniforms(const mat4& view, const mat4 &proj);
    void render(const mat4& view, const mat4 &proj);
};
