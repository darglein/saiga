/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/deferredRendering/lighting/shadowmap.h"

#include "saiga/opengl/error.h"
#include "saiga/opengl/texture/CubeTexture.h"

namespace Saiga
{
void ShadowmapBase::bindFramebuffer()
{
#if defined(SAIGA_DEBUG)
    depthBuffer.check();
#endif
    glViewport(0, 0, w, h);
    depthBuffer.bind();
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}

void ShadowmapBase::unbindFramebuffer()
{
    depthBuffer.unbind();
}


SimpleShadowmap::SimpleShadowmap(int w, int h, ShadowQuality quality)
{
    this->w = w;
    this->h = h;
    depthBuffer.create();
    depthBuffer.unbind();

    std::shared_ptr<Texture> depth = std::make_shared<Texture>();

    switch (quality)
    {
        case ShadowQuality::LOW:
            depth->create(w, h, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16, GL_UNSIGNED_SHORT);
            break;
        case ShadowQuality::MEDIUM:
            depth->create(w, h, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32, GL_UNSIGNED_INT);
            break;
        case ShadowQuality::HIGH:
            depth->create(w, h, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32F, GL_FLOAT);
            break;
    }

    //    depth->create(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
    //    depth->setWrap(GL_CLAMP_TO_EDGE);
    depth->setWrap(GL_CLAMP_TO_BORDER);
    //    depth->setBorderColor(vec4(1.0f));
    depth->setBorderColor(make_vec4(0.0f));  // no light on the outside
    depth->setFiltering(GL_LINEAR);

    // this requires the texture sampler in the shader to be sampler2DShadow
    depth->setParameter(GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    depth->setParameter(GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

    depthTexture = depth;

    depthBuffer.attachTextureDepth(depth);
    depthBuffer.check();

    //    initialized = true;

    assert_no_glerror();
}


CubeShadowmap::CubeShadowmap(int w, int h, ShadowQuality quality)
{
    this->w = w;
    this->h = h;
    depthBuffer.create();
    depthBuffer.unbind();


    auto cubeMap = std::make_shared<TextureCube>();
    //    cubeMap->create(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
    switch (quality)
    {
        case ShadowQuality::LOW:
            cubeMap->create(w, h, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16, GL_UNSIGNED_SHORT);
            break;
        case ShadowQuality::MEDIUM:
            cubeMap->create(w, h, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32, GL_UNSIGNED_INT);
            break;
        case ShadowQuality::HIGH:
            cubeMap->create(w, h, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32F, GL_FLOAT);
            break;
    }

    //    cubeMap->create(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
    cubeMap->setWrap(GL_CLAMP_TO_EDGE);
    cubeMap->setFiltering(GL_LINEAR);

    // this requires the texture sampler in the shader to be samplerCubeShadow
    cubeMap->setParameter(GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    cubeMap->setParameter(GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

    depthTexture = cubeMap;
    //    deleteTexture = cubeMap;
    //    initialized = true;

    assert_no_glerror();
}

void CubeShadowmap::bindCubeFace(GLenum side)
{
    glViewport(0, 0, w, h);

    depthBuffer.bind();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, side, getDepthTexture()->getId(), 0);
    //    depthBuffer.drawToNone();


    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

#if defined(SAIGA_DEBUG)
    depthBuffer.check();
#endif
}


CascadedShadowmap::CascadedShadowmap(int w, int h, int numCascades, ShadowQuality quality)
{
    this->w = w;
    this->h = h;
    depthBuffer.create();
    depthBuffer.unbind();



#if 0
    for(int i = 0 ; i < numCascades; ++i){
        std::shared_ptr<Texture> depth = std::make_shared<Texture>();

        switch(quality){
        case ShadowQuality::LOW:
            depth->create(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
            break;
        case ShadowQuality::MEDIUM:
            depth->create(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
            break;
        case ShadowQuality::HIGH:
            depth->create(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32F,GL_FLOAT);
            break;
        }

        //    depth->create(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
        //    depth->setWrap(GL_CLAMP_TO_EDGE);
        depth->setWrap(GL_CLAMP_TO_BORDER);
        depth->setBorderColor(vec4(1.0f));
        depth->setFiltering(GL_LINEAR);

        //this requires the texture sampler in the shader to be sampler2DShadow
        depth->setParameter(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_REF_TO_TEXTURE);
        depth->setParameter(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL);

        depthTextures.push_back(depth);
    }
#endif

    depthTexture = std::make_shared<ArrayTexture2D>();

    switch (quality)
    {
        case ShadowQuality::LOW:
            depthTexture->create(w, h, numCascades, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16, GL_UNSIGNED_SHORT);
            break;
        case ShadowQuality::MEDIUM:
            depthTexture->create(w, h, numCascades, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32, GL_UNSIGNED_INT);
            break;
        case ShadowQuality::HIGH:
            depthTexture->create(w, h, numCascades, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32F, GL_FLOAT);
            break;
    }
    depthTexture->setWrap(GL_CLAMP_TO_BORDER);
    depthTexture->setBorderColor(make_vec4(1.0f));
    depthTexture->setFiltering(GL_LINEAR);

    // this requires the texture sampler in the shader to be sampler2DShadow
    depthTexture->setParameter(GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    depthTexture->setParameter(GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

    assert_no_glerror();
}

void CascadedShadowmap::bindAttachCascade(int n)
{
    glViewport(0, 0, w, h);

    depthBuffer.bind();
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture->getId(), 0, n);
    //    depthBuffer.attachTextureDepth(getDepthTexture(n));
    //    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, side, getDepthTexture(0)->getId(), 0);
    //    depthBuffer.drawToNone();


    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

#if defined(SAIGA_DEBUG)
    depthBuffer.check();
#endif
}


}  // namespace Saiga
