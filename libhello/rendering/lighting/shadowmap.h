#pragma once

#include "libhello/opengl/mesh_object.h"
#include "libhello/opengl/basic_shaders.h"
#include "libhello/geometry/sphere.h"
#include "libhello/geometry/plane.h"
#include "libhello/geometry/triangle_mesh.h"
#include "libhello/opengl/framebuffer.h"
#include "libhello/opengl/texture/cube_texture.h"

class Shadowmap{
private:
    bool initialized = false;
public:
    int w,h;
    Framebuffer depthBuffer;
    raw_Texture* depthTexture = nullptr;



    void bind();
    void unbind();
    void bindCubeFace(GLenum side);

    bool isInitialized(){ return initialized;}
    void init(int w, int h);
    void createFlat(int w, int h);
    void createCube(int w, int h);
};

