/**
 * Copyright (c) 2021 Darius Rückert
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

class SAIGA_OPENGL_API MVPShader : public Shader
{
   public:
    GLint location_model;
    GLint location_cameraData;
    GLint location_userData;

    virtual void checkUniforms();

    void uploadModel(const mat4& matrix) { upload(location_model, matrix); }
    void uploadUserData(float f) { upload(location_userData, f); }
};

class SAIGA_OPENGL_API MVPColorShader : public MVPShader
{
   public:
    GLint location_color;

    virtual void checkUniforms();
    virtual void uploadColor(const vec4& color);
};

class SAIGA_OPENGL_API MVPTextureShader : public MVPShader
{
   public:
    GLint location_texture;

    virtual void checkUniforms();
    virtual void uploadTexture(TextureBase* texture);
};


// class SAIGA_OPENGL_API FBShader : public MVPShader
//{
//   public:
//    GLint location_texture;

//    virtual void checkUniforms();
//    virtual void uploadFramebuffer(Framebuffer* fb);
//};

class SAIGA_OPENGL_API DeferredShader : public MVPShader
{
   public:
    GLint location_viewPort;
    GLint location_texture_diffuse, location_texture_normal, location_texture_depth, location_texture_data;

    virtual void checkUniforms();
    void uploadFramebuffer(GBuffer* gbuffer);
    void uploadScreenSize(const vec4& vp) { Shader::upload(location_viewPort, vp); }
};

}  // namespace Saiga
