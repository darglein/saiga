#pragma once

#include "saiga/opengl/texture/texture.h"
#include "saiga/opengl/texture/cube_texture.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/camera/camera.h"



class SAIGA_GLOBAL ProceduralSkybox{
public:
    IndexedVertexBuffer<VertexNT,GLuint> mesh;
    MVPShader* shader;
    mat4 model;

    ProceduralSkybox();

    void render(Camera *cam);
};
