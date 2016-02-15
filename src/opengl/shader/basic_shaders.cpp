#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/rendering/gbuffer.h"
void MVPColorShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_color = getUniformLocation("color");
}

void MVPColorShader::uploadColor(const vec4 &color){
    upload(location_color,color);
}

void MVPTextureShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_texture = Shader::getUniformLocation("image");
}


void MVPTextureShader::uploadTexture(raw_Texture *texture){
    upload(location_texture,texture,0);
}



void FBShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_texture = getUniformLocation("text");
}

void FBShader::uploadFramebuffer(Framebuffer *fb){
//    fb->colorBuffers[0]->bind(0);
//    upload(location_texture,0);
//    upload(location_texture,fb->colorBuffers[0],0);
}

void DeferredShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_screen_size = getUniformLocation("screen_size");

    location_texture_diffuse = getUniformLocation("deferred_diffuse");
    location_texture_normal = getUniformLocation("deferred_normal");
    location_texture_depth = getUniformLocation("deferred_depth");
    location_texture_data = getUniformLocation("deferred_data");

}

void DeferredShader::uploadFramebuffer(GBuffer *gbuffer){
//    upload(location_texture_diffuse,fb->colorBuffers[0],0);
//    upload(location_texture_normal,fb->colorBuffers[1],1);
//    upload(location_texture_data,fb->colorBuffers[2],2);
//    upload(location_texture_depth,fb->depthBuffer,3);
    upload(location_texture_diffuse,gbuffer->getTextureColor(),0);
    upload(location_texture_normal,gbuffer->getTextureNormal(),1);
    upload(location_texture_data,gbuffer->getTextureData(),2);
    upload(location_texture_depth,gbuffer->getTextureDepth(),3);
}

void MVPShader::checkUniforms(){
    Shader::checkUniforms();
    location_model = getUniformLocation("model");
    location_view = getUniformLocation("view");
    location_proj = getUniformLocation("proj");
    location_mv = getUniformLocation("MV");
    location_mvp = getUniformLocation("MVP");
}

void MVPShader::uploadAll(const mat4& m1,const mat4& m2,const mat4& m3){
    uploadModel(m1);
    uploadView(m2);
    uploadProj(m3);

    uploadMV(m2*m1);
    uploadMVP(m3*m2*m1);
}

