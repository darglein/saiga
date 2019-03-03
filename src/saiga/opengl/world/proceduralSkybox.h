/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/texture/cube_texture.h"
#include "saiga/opengl/texture/texture.h"
#include "saiga/opengl/vertex.h"

namespace Saiga
{
class SAIGA_OPENGL_API ProceduralSkyboxShader : public MVPShader
{
   public:
    GLint location_params;


    virtual void checkUniforms();
    virtual void uploadParams(vec3 sunDir, float horizonHeight, float distance, float sunIntensity, float sunSize);
};

class SAIGA_OPENGL_API ProceduralSkybox
{
   public:
    float horizonHeight = 0;
    float distance      = 200;
    float sunIntensity  = 1;
    float sunSize       = 1;
    vec3 sunDir         = vec3(0, -1, 0);

    IndexedVertexBuffer<VertexNT, GLuint> mesh;
    std::shared_ptr<ProceduralSkyboxShader> shader;
    mat4 model;

    ProceduralSkybox();

    void render(Camera* cam);

    void imgui();
};

}  // namespace Saiga
