#include "rendering/lighting/shadowmap.h"

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
    depth->setWrap(GL_CLAMP_TO_EDGE);
    depthTexture = depth;

    depthBuffer.attachTextureDepth(depth);
    depthBuffer.check();

    initialized = true;
}

void Shadowmap::createCube(int w, int h){
    init(w,h);



    cube_Texture* cubeMap = new cube_Texture();
    cubeMap->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
    depthTexture = cubeMap;

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
