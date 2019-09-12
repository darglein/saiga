/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/texture/ArrayTexture2D.h"

namespace Saiga
{
enum class ShadowQuality
{
    LOW,     // 16 uint
    MEDIUM,  // 32 uint
    HIGH     // 32 bit float
};


class SAIGA_OPENGL_API ShadowmapBase
{
   protected:
    int w, h;
    Framebuffer depthBuffer;

   public:
    ivec2 getSize() { return ivec2(w, h); }
    void bindFramebuffer();
    void unbindFramebuffer();
};

/**
 * Simple shadow map with one 2D depth texture.
 * Used by box- and spotlight
 */
class SAIGA_OPENGL_API SimpleShadowmap : public ShadowmapBase
{
    std::shared_ptr<TextureBase> depthTexture;

   public:
    SimpleShadowmap(int w, int h, ShadowQuality quality = ShadowQuality::LOW);
    ~SimpleShadowmap() {}
    std::shared_ptr<TextureBase> getDepthTexture() { return depthTexture; }
};

/**
 * Cube shadow map with one cube depth texture.
 * Used by point light
 */
class SAIGA_OPENGL_API CubeShadowmap : public ShadowmapBase
{
    std::shared_ptr<TextureBase> depthTexture;

   public:
    CubeShadowmap(int w, int h, ShadowQuality quality = ShadowQuality::LOW);
    ~CubeShadowmap() {}
    std::shared_ptr<TextureBase> getDepthTexture() { return depthTexture; }
    void bindCubeFace(GLenum side);
};

/**
 * Cascaded shadow map with numCascades depth textures.
 * Used by directional light
 */
class SAIGA_OPENGL_API CascadedShadowmap : public ShadowmapBase
{
    //    std::vector<std::shared_ptr<raw_Texture>> depthTextures;
    std::shared_ptr<ArrayTexture2D> depthTexture;

   public:
    CascadedShadowmap(int w, int h, int numCascades, ShadowQuality quality = ShadowQuality::LOW);
    ~CascadedShadowmap() {}

    //    std::shared_ptr<raw_Texture> getDepthTexture(unsigned int n){
    //        SAIGA_ASSERT(n < depthTextures.size());
    //        return depthTextures[n];
    //    }
    //    std::vector<std::shared_ptr<raw_Texture>>& getDepthTextures(){ return depthTextures;}

    std::shared_ptr<ArrayTexture2D> getDepthTexture() { return depthTexture; }

    void bindAttachCascade(int n);
};

}  // namespace Saiga
