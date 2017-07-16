#pragma once

#include <saiga/config.h>

namespace Saiga {

class Camera;
class OpenGLWindow;

class SAIGA_GLOBAL Program{
public:
    OpenGLWindow* parentWindow = nullptr;

    Program(OpenGLWindow* parent);

    virtual ~Program(){}

    //advances the state of the program by dt. All game logic should happen here
    virtual void update(float dt) = 0;

    virtual void parallelUpdate(float dt){}

    //interpolation between two logic steps for high fps rendering.
    //Example:
    // Game loop: constant 60 Hz
    // Render rate: around 120 Hz
    //-> The game is rendered two times per update
    //
    //We don't want to render two times the same image, so the game state should be interpolated either into the future or from the past.
    //Alpha is in the range [0,1] where 1 is equivalent to a timestep of dt
    virtual void interpolate(float dt, float alpha)  = 0;

    //rendering into the gbuffer
    virtual void render(Camera *cam)  = 0;

    //render depth maps for shadow lights
    virtual void renderDepth(Camera *cam)  = 0;

    //forward rendering path after lighting, but before post processing
    //this could be used for transparent objects
    virtual void renderOverlay(Camera *cam)  = 0;

    //forward rendering path after lighting and after post processing
    //typical used for the gui
    virtual void renderFinal(Camera *cam)  = 0;
};

}
