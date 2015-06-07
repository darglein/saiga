#include "rendering/lighting/shadowmap.h"

Shadowmap::Shadowmap()
{

}

Shadowmap::~Shadowmap(){
//    delete depthTexture;
    delete deleteTexture;
}

void Shadowmap::init(int w, int h){
    this->w = w;
    this->h = h;

    depthBuffer.destroy();

    depthBuffer.create();
    depthBuffer.unbind();

    delete depthTexture;
    depthTexture = nullptr;

}

void Shadowmap::createFlat(int w, int h){
    init(w,h);

    Texture* depth = new Texture();
    depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
//    depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
    depth->setWrap(GL_CLAMP_TO_EDGE);
    depth->setFiltering(GL_LINEAR);

     //this requires the texture sampler in the shader to be sampler2DShadow
    depth->setParameter(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_REF_TO_TEXTURE);
    depth->setParameter(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL);
    depthTexture = depth;

    depthBuffer.attachTextureDepth(depth);
    depthBuffer.check();

    initialized = true;
}

void Shadowmap::createCube(int w, int h){
    init(w,h);



    cube_Texture* cubeMap = new cube_Texture();
    cubeMap->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
//    cubeMap->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
    cubeMap->setWrap(GL_CLAMP_TO_EDGE);
    cubeMap->setFiltering(GL_LINEAR);

    //this requires the texture sampler in the shader to be samplerCubeShadow
    cubeMap->setParameter(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_REF_TO_TEXTURE);
    cubeMap->setParameter(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL);

    depthTexture = cubeMap;
    deleteTexture = cubeMap;
    initialized = true;

}




void Shadowmap::bind(){
    glViewport(0,0,w,h);
    depthBuffer.bind();
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}

void Shadowmap::unbind(){
    depthBuffer.unbind();
}

void Shadowmap::bindCubeFace(GLenum side){
    glViewport(0,0,w,h);

    depthBuffer.bind();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, side, depthTexture->getId(), 0);
    glDrawBuffer(GL_NONE);



    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}
