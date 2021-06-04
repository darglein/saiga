/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/shader/basic_shaders.h"

#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/rendering/deferredRendering/gbuffer.h"

namespace Saiga
{
void MVPShader::checkUniforms()
{
    Shader::checkUniforms();
    location_model    = getUniformLocation("model");
    location_userData = getUniformLocation("userData");

    location_cameraData = getUniformBlockLocation("cameraData");
    if (location_cameraData != GL_INVALID_INDEX) setUniformBlockBinding(location_cameraData, CAMERA_DATA_BINDING_POINT);
}



void MVPColorShader::checkUniforms()
{
    MVPShader::checkUniforms();
    location_color = getUniformLocation("color");
}

void MVPColorShader::uploadColor(const vec4& color)
{
    upload(location_color, color);
}

void MVPTextureShader::checkUniforms()
{
    MVPShader::checkUniforms();
    location_texture = Shader::getUniformLocation("image");
}


void MVPTextureShader::uploadTexture(TextureBase* texture)
{
    upload(location_texture, texture, 0);
}



// void FBShader::checkUniforms()
//{
//    MVPShader::checkUniforms();
//    location_texture = getUniformLocation("text");
//}

// void FBShader::uploadFramebuffer(Framebuffer* fb)
//{
//    //    fb->colorBuffers[0]->bind(0);
//    //    upload(location_texture,0);
//    //    upload(location_texture,fb->colorBuffers[0],0);
//}

void DeferredShader::checkUniforms()
{
    MVPShader::checkUniforms();
    location_viewPort = getUniformLocation("viewPort");

    location_texture_diffuse = getUniformLocation("deferred_diffuse");
    location_texture_normal  = getUniformLocation("deferred_normal");
    location_texture_depth   = getUniformLocation("deferred_depth");
    location_texture_data    = getUniformLocation("deferred_data");
}

void DeferredShader::uploadFramebuffer(GBuffer* gbuffer)
{
    //    upload(location_texture_diffuse,fb->colorBuffers[0],0);
    //    upload(location_texture_normal,fb->colorBuffers[1],1);
    //    upload(location_texture_data,fb->colorBuffers[2],2);
    //    upload(location_texture_depth,fb->depthBuffer,3);
    upload(location_texture_diffuse, gbuffer->getTextureColor(), 0);
    upload(location_texture_normal, gbuffer->getTextureNormal(), 1);
    upload(location_texture_data, gbuffer->getTextureMaterial(), 2);
    upload(location_texture_depth, gbuffer->getTextureDepth(), 3);
}

}  // namespace Saiga
