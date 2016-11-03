#pragma once

#include "saiga/camera/controllable_camera.h"
#include <SDL2/SDL.h>

template<typename camera_t>
class SDLCamera : public Controllable_Camera<camera_t>{
public:
    SDLCamera(){
        this->keyboardmap = {
//            SDL_SCANCODE_UP,
//            SDL_SCANCODE_DOWN,
//            SDL_SCANCODE_LEFT,
//            SDL_SCANCODE_RIGHT,
            SDL_SCANCODE_W,
            SDL_SCANCODE_S,
            SDL_SCANCODE_A,
            SDL_SCANCODE_D,
            SDL_SCANCODE_LSHIFT,
            SDL_SCANCODE_SPACE
        };
        this->mousemap = {
            SDL_BUTTON_MIDDLE
        };
    }
};


