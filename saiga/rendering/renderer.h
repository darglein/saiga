#pragma once

#include <saiga/config.h>

class Camera;

class SAIGA_GLOBAL RendererInterface{
public:
    virtual ~RendererInterface(){}
    virtual void update(float dt) = 0;
    virtual void interpolate(float interpolation)  = 0;
    virtual void render(Camera *cam)  = 0;
    virtual void renderDepth(Camera *cam)  = 0;
    virtual void renderOverlay(Camera *cam)  = 0;
    virtual void renderFinal(Camera *cam)  = 0;
};
