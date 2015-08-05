#pragma once

#include <saiga/config.h>
#include "saiga/window/window.h"

#include "saiga/sdl/sdl_eventhandler.h"
#include <SDL2/SDL.h>
//#include <SDL2/SDL_opengl.h>


class SAIGA_GLOBAL sdl_Window : public Window{
protected:
    SDL_Window* gWindow = NULL;
    SDL_GLContext gContext;

    bool initWindow();
    bool initInput();
public:
    SDL_EventHandler eventHandler;

    sdl_Window(const std::string &name,int width,int height);


    void close();
    void startMainLoop();
};

