/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/shader/shader.h"

namespace Saiga
{
class Camera;
class Framebuffer;
class GBuffer;

#define CAMERA_DATA_BINDING_POINT 0

class SAIGA_GLOBAL MVPShader : public Shader
{
   public:
    GLint location_model;
    GLint location_cameraData;
    GLint location_userData;

    virtual void checkUniforms();

    void uploadModel(const mat4& matrix) { upload(location_model, matrix); }
    void uploadUserData(float f) { upload(location_userData, f); }
};

class SAIGA_GLOBAL MVPColorShader : public MVPShader
{
   public:
    GLint location_color;

    virtual void checkUniforms();
    virtual void uploadColor(const vec4& color);
};

class SAIGA_GLOBAL MVPTextureShader : public MVPShader
{
   public:
    GLint location_texture;

    virtual void checkUniforms();
    virtual void uploadTexture(std::shared_ptr<raw_Texture> texture);
};


class SAIGA_GLOBAL FBShader : public MVPShader
{
   public:
    GLint location_texture;

    virtual void checkUniforms();
    virtual void uploadFramebuffer(Framebuffer* fb);
};

class SAIGA_GLOBAL DeferredShader : public MVPShader
{
   public:
    GLint location_screen_size;
    GLint location_texture_diffuse, location_texture_normal, location_texture_depth, location_texture_data;

    virtual void checkUniforms();
    void uploadFramebuffer(GBuffer* gbuffer);
    void uploadScreenSize(vec2 sc) { Shader::upload(location_screen_size, sc); }
};

}  // namespace Saiga
