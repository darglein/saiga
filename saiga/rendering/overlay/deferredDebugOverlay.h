#pragma once

#include "saiga/util/glm.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/rendering/object3d.h"
#include <vector>

class MVPTextureShader;
class basic_Texture_2D;
class Framebuffer;




class SAIGA_GLOBAL DeferredDebugOverlay {
private:
    struct GbufferTexture : public Object3D{
        basic_Texture_2D *texture;
    };


    mat4 proj;


    int width,height;

    GbufferTexture color, normal, depth, data, light;

    void setScreenPosition(GbufferTexture* gbt, int id);
public:

    MVPTextureShader* shader, *depthShader, *normalShader;
    IndexedVertexBuffer<VertexNT,GLuint> buffer;

    DeferredDebugOverlay(int width, int height);
    void render();

    void setDeferredFramebuffer(Framebuffer* fb, basic_Texture_2D *light);


};


