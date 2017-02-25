#pragma once


#include "saiga/opengl/framebuffer.h"

class SAIGA_GLOBAL Shadowmap{
private:
    bool initialized = false;
public:
    int w,h;
    Framebuffer depthBuffer;
    std::shared_ptr<raw_Texture> depthTexture;
    std::shared_ptr<raw_Texture> deleteTexture;

    Shadowmap();
    ~Shadowmap();
    void bind();
    void unbind();
    void bindCubeFace(GLenum side);

    bool isInitialized(){ return initialized;}
    void init(int w, int h);
    void createFlat(int w, int h);
    void createCube(int w, int h);
};

