/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/object3d.h"
#include "saiga/core/math/math.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/rendering/overlay/Layout.h"
#include "saiga/opengl/vertex.h"

#include <memory>
namespace Saiga
{
class MVPTextureShader;
class basic_Texture_2D;
class TextureBase;
class Framebuffer;
class GBuffer;



class SAIGA_OPENGL_API DeferredDebugOverlay
{
   public:
    Layout layout;

    DeferredDebugOverlay(int width, int height);

    void render();

    void setDeferredFramebuffer(GBuffer* gbuffer, std::shared_ptr<TextureBase> light);

   private:
    struct GbufferTexture : public Object3D
    {
        std::shared_ptr<TextureBase> texture;
    };


    AABB meshBB;


    GbufferTexture color, normal, depth, data, light;

    std::shared_ptr<MVPTextureShader> shader, depthShader, normalShader;
    IndexedVertexBuffer<VertexNT, GLuint> buffer;

    void setScreenPosition(GbufferTexture* gbt, int id);
    void loadShaders();
};

}  // namespace Saiga
