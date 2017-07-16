#include "saiga/rendering/lighting/shadowmap.h"
#include "saiga/opengl/texture/cube_texture.h"
#include "saiga/util/error.h"

namespace Saiga {

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

    depthTextures.clear();

    //    delete depthTexture;
    //    depthTexture = nullptr;

}

void Shadowmap::createFlat(int w, int h, ShadowQuality quality){
    init(w,h);

    std::shared_ptr<Texture> depth = std::make_shared<Texture>();

    switch(quality){
    case LOW:
        depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
        break;
    case MEDIUM:
        depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
        break;
    case HIGH:
        depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32F,GL_FLOAT);
        break;
    }

    //    depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
    //    depth->setWrap(GL_CLAMP_TO_EDGE);
    depth->setWrap(GL_CLAMP_TO_BORDER);
    depth->setBorderColor(vec4(1.0f));
    depth->setFiltering(GL_LINEAR);

    //this requires the texture sampler in the shader to be sampler2DShadow
    depth->setParameter(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_REF_TO_TEXTURE);
    depth->setParameter(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL);

    depthTextures.push_back(depth);

    depthBuffer.attachTextureDepth( depth );
    depthBuffer.check();

    initialized = true;

    assert_no_glerror();
}

void Shadowmap::createCube(int w, int h, ShadowQuality quality){
    init(w,h);



    auto cubeMap = std::make_shared<TextureCube>();
    //    cubeMap->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
    switch(quality){
    case LOW:
        cubeMap->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
        break;
    case MEDIUM:
        cubeMap->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
        break;
    case HIGH:
        cubeMap->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32F,GL_FLOAT);
        break;
    }

    //    cubeMap->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
    cubeMap->setWrap(GL_CLAMP_TO_EDGE);
    cubeMap->setFiltering(GL_LINEAR);

    //this requires the texture sampler in the shader to be samplerCubeShadow
    cubeMap->setParameter(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_REF_TO_TEXTURE);
    cubeMap->setParameter(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL);

    depthTextures.push_back(cubeMap);
    //    deleteTexture = cubeMap;
    initialized = true;

    assert_no_glerror();
}


void Shadowmap::createCascaded(int w, int h, int numCascades, ShadowQuality quality){
    init(w,h);


    for(int i = 0 ; i < numCascades; ++i){
        std::shared_ptr<Texture> depth = std::make_shared<Texture>();

        switch(quality){
        case LOW:
            depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
            break;
        case MEDIUM:
            depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
            break;
        case HIGH:
            depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32F,GL_FLOAT);
            break;
        }

        //    depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
        //    depth->setWrap(GL_CLAMP_TO_EDGE);
        depth->setWrap(GL_CLAMP_TO_BORDER);
        depth->setBorderColor(vec4(1.0f));
        depth->setFiltering(GL_LINEAR);

        //this requires the texture sampler in the shader to be sampler2DShadow
        depth->setParameter(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_REF_TO_TEXTURE);
        depth->setParameter(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL);

        depthTextures.push_back(depth);
    }

//    depthBuffer.attachTextureDepth( depth );
//    depthBuffer.check();

    initialized = true;

    assert_no_glerror();
}



void Shadowmap::bindFramebuffer(){
    glViewport(0,0,w,h);
    depthBuffer.bind();
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}

void Shadowmap::unbindFramebuffer(){
    depthBuffer.unbind();
}

void Shadowmap::bindCubeFace(GLenum side){
    glViewport(0,0,w,h);

    depthBuffer.bind();
//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, side, getDepthTexture(0)->getId(), 0);
//    depthBuffer.drawToNone();


    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

#if defined(SAIGA_DEBUG)
    depthBuffer.check();
#endif
}

void Shadowmap::bindAttachCascade(int n){
    glViewport(0,0,w,h);

    depthBuffer.bind();
    depthBuffer.attachTextureDepth(getDepthTexture(n));
//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, side, getDepthTexture(0)->getId(), 0);
//    depthBuffer.drawToNone();


    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

//#if defined(SAIGA_DEBUG)
    depthBuffer.check();
//#endif
}

}
