#pragma once

#include <saiga/config.h>
#include "saiga/window/window.h"

#include "saiga/sdl/sdl_eventhandler.h"
#include <SDL2/SDL.h>
//#include <SDL2/SDL_opengl.h>


class SAIGA_GLOBAL SDLWindow : public OpenGLWindow{
public:

    SDL_Window* window = NULL;
protected:
    SDL_GLContext gContext;

    virtual bool initWindow() override;
    virtual bool initInput() override;
    virtual bool shouldClose() override;
    virtual void checkEvents() override;
    virtual void swapBuffers() override;
    virtual void freeContext() override;
public:

    SDLWindow(WindowParameters windowParameters);
};

