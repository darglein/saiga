#include "saiga/rendering/lighting/shadowmap.h"

#include "saiga/opengl/texture/cube_texture.h"
#include "saiga/util/error.h"
Shadowmap::Shadowmap()
{

}

Shadowmap::~Shadowmap(){
//    delete depthTexture;
//    delete deleteTexture;
}

void Shadowmap::init(int w, int h){
    this->w = w;
    this->h = h;

    depthBuffer.destroy();

    depthBuffer.create();
    depthBuffer.unbind();

//    delete depthTexture;
//    depthTexture = nullptr;

}

void Shadowmap::createFlat(int w, int h){
    init(w,h);

    std::shared_ptr<Texture> depth = std::make_shared<Texture>();;
    depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
//    depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
//    depth->setWrap(GL_CLAMP_TO_EDGE);
    depth->setWrap(GL_CLAMP_TO_BORDER);
    depth->setBorderColor(vec4(1.0f));
    depth->setFiltering(GL_LINEAR);

     //this requires the texture sampler in the shader to be sampler2DShadow
    depth->setParameter(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_REF_TO_TEXTURE);
    depth->setParameter(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL);
    depthTexture = depth;

    depthBuffer.attachTextureDepth( framebuffer_texture_t(depth) );
    depthBuffer.check();

    initialized = true;

    assert_no_glerror();
}

void Shadowmap::createCube(int w, int h){
    init(w,h);



    auto cubeMap = std::make_shared<TextureCube>();
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

    assert_no_glerror();
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
    depthBuffer.drawToNone();


    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}
