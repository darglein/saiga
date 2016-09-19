#pragma once

#include "saiga/opengl/vertex.h"
#include "saiga/opengl/texture/texture.h"
#include "saiga/opengl/texture/cube_texture.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"




class SAIGA_GLOBAL Skybox{
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
