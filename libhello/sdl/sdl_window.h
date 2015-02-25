#pragma once

#include "libhello/window/window.h"

#include "libhello/sdl/sdl_eventhandler.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>


class sdl_Window : public Window{
protected:
    SDL_Window* gWindow = NULL;
    SDL_GLContext gContext;

    bool initWindow();
    bool initInput();
public:
    SDL_EventHandler eventHandler;

    sdl_Window(const std::string &name,int window_width,int window_height);


    void close();
    void startMainLoop();
};

