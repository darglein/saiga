#pragma once

#include "saiga/opengl/framebuffer.h"
#include "saiga/util/assert.h"

namespace Saiga {

class SAIGA_GLOBAL Shadowmap{
private:
    bool initialized = false;
    int w,h;

    Framebuffer depthBuffer;

    std::vector<std::shared_ptr<raw_Texture>> depthTextures;
//    std::shared_ptr<raw_Texture> depthTexture;
public:
//    std::shared_ptr<raw_Texture> deleteTexture;

    Shadowmap();
    ~Shadowmap();

    std::shared_ptr<raw_Texture> getDepthTexture(unsigned int n){
        SAIGA_ASSERT(n < depthTextures.size());
        return depthTextures[n];
    }

    std::vector<std::shared_ptr<raw_Texture>>& getDepthTextures(){ return depthTextures;}

    void bindFramebuffer();
    void unbindFramebuffer();

    glm::ivec2 getSize(){ return glm::ivec2(w,h);}
    void bindCubeFace(GLenum side);
    void bindAttachCascade(int n);

    bool isInitialized(){ return initialized;}
    void init(int w, int h);

    enum ShadowQuality{
        LOW,        // 16 uint
        MEDIUM,     // 32 uint
        HIGH        // 32 bit float
    };

    void createFlat(int w, int h, ShadowQuality quality = LOW);
    void createCube(int w, int h, ShadowQuality quality = LOW);
    void createCascaded(int w, int h, int numCascades, ShadowQuality quality = LOW);
};

}
