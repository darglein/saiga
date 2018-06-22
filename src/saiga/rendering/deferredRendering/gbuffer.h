/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/quality.h"
#include "saiga/opengl/framebuffer.h"

namespace Saiga {

struct SAIGA_GLOBAL GBufferParameters{
    bool srgb = false; //colors stored in srgb. saves memory bandwith but adds conversion operations.
    Quality colorQuality = Quality::MEDIUM;
    Quality normalQuality = Quality::MEDIUM;
    Quality dataQuality = Quality::LOW;
    Quality depthQuality = Quality::HIGH;
};

class SAIGA_GLOBAL GBuffer : public Framebuffer{
protected:
    GBufferParameters params;
public:
    GBuffer();
    GBuffer(int w, int h, GBufferParameters params);
    void init(int w, int h, GBufferParameters params);

    framebuffer_texture_t getTextureColor(){return this->colorBuffers[0];}
    framebuffer_texture_t getTextureNormal(){return this->colorBuffers[1];}
    framebuffer_texture_t getTextureData(){return this->colorBuffers[2];}

    void sampleNearest();
    void sampleLinear();

    void clampToEdge();
};

}
