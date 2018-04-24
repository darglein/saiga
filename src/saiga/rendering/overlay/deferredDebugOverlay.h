/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/glm.h"
#include "saiga/opengl/vertex.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/rendering/object3d.h"
#include "saiga/rendering/overlay/Layout.h"

namespace Saiga {

class MVPTextureShader;
class basic_Texture_2D;
class raw_Texture;
class Framebuffer;
class GBuffer;



class SAIGA_GLOBAL DeferredDebugOverlay {
private:
    struct GbufferTexture : public Object3D{
        std::shared_ptr<raw_Texture> texture;
    };


    AABB meshBB;

    int width,height;

    GbufferTexture color, normal, depth, data, light;

    void setScreenPosition(GbufferTexture* gbt, int id);
public:
    Layout layout;

    std::shared_ptr<MVPTextureShader>  shader, depthShader, normalShader;
    IndexedVertexBuffer<VertexNT,GLuint> buffer;

    DeferredDebugOverlay(int width, int height);

    void loadShaders();
    void render();

    void setDeferredFramebuffer(GBuffer *gbuffer, std::shared_ptr<raw_Texture> light);


};

}
