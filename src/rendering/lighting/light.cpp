#include "rendering/lighting/light.h"

void LightShader::checkUniforms(){
    DeferredShader::checkUniforms();
    location_color = getUniformLocation("color");
    location_depthBiasMV = getUniformLocation("depthBiasMV");
    location_depthTex = getUniformLocation("depthTex");
}


void LightShader::uploadColor(vec4 &color){
    Shader::upload(location_color,color);
}

void LightShader::uploadColor(vec3 &color, float intensity){
    vec4 c = vec4(color,intensity);
    Shader::upload(location_color,c);
}

void LightShader::uploadDepthBiasMV(mat4 &mat){
    Shader::upload(location_depthBiasMV,mat);
}

void LightShader::uploadDepthTexture(raw_Texture* texture){

        texture->bind(4);
        Shader::upload(location_depthTex,4);
}

void Light::createShadowMap(int resX, int resY){
    shadowResX = resX;
    shadowResY = resY;

    depthBuffer.create();
    Texture* depth = new Texture();
    depth->createEmptyTexture(shadowResX,shadowResY,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
    depth->setWrap(GL_CLAMP_TO_EDGE);
    depthBuffer.attachTextureDepth(depth);
    depthBuffer.check();
}

void Light::bindShadowMap(){
    glViewport(0,0,shadowResX,shadowResY);
    depthBuffer.bind();
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}

void Light::unbindShadowMap(){
    depthBuffer.unbind();
}
