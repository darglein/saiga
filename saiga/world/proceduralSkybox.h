#pragma once

#include "saiga/opengl/vertex.h"
#include "saiga/opengl/texture/texture.h"
#include "saiga/opengl/texture/cube_texture.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/camera/camera.h"

namespace Saiga {

class SAIGA_GLOBAL ProceduralSkyboxShader : public MVPShader{
public:
    GLint location_params;


    virtual void checkUniforms();
    virtual void uploadParams(float horizonHeight, float distance);
};

class SAIGA_GLOBAL ProceduralSkybox{
public:
    float horizonHeight = 0;
    float distance = 200;

    IndexedVertexBuffer<VertexNT,GLuint> mesh;
    std::shared_ptr<ProceduralSkyboxShader>  shader;
    mat4 model;

    ProceduralSkybox();

    void render(Camera *cam);
};

}
