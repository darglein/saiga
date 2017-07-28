/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/framebuffer.h"
#include "saiga/util/assert.h"

namespace Saiga {

enum class ShadowQuality{
    LOW,        // 16 uint
    MEDIUM,     // 32 uint
    HIGH        // 32 bit float
};


class SAIGA_GLOBAL ShadowmapBase{
protected:
    bool initialized = false;
    int w,h;
    Framebuffer depthBuffer;
public:
    void bindFramebuffer();
    void unbindFramebuffer();
    bool isInitialized(){ return initialized;}
};

class SAIGA_GLOBAL Shadowmap : public ShadowmapBase{

    std::vector<std::shared_ptr<raw_Texture>> depthTextures;

public:

    Shadowmap(){}
    ~Shadowmap(){}

    std::shared_ptr<raw_Texture> getDepthTexture(unsigned int n){
        SAIGA_ASSERT(n < depthTextures.size());
        return depthTextures[n];
    }

    std::vector<std::shared_ptr<raw_Texture>>& getDepthTextures(){ return depthTextures;}


    glm::ivec2 getSize(){ return glm::ivec2(w,h);}
    void bindCubeFace(GLenum side);
    void bindAttachCascade(int n);

    void init(int w, int h);


    void createFlat(int w, int h, ShadowQuality quality = ShadowQuality::LOW);
    void createCube(int w, int h, ShadowQuality quality = ShadowQuality::LOW);
    void createCascaded(int w, int h, int numCascades, ShadowQuality quality = ShadowQuality::LOW);
};

}
