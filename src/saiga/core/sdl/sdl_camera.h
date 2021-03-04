/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/controllable_camera.h"
#include "saiga/core/sdl/saiga_sdl.h"

namespace Saiga
{
template <typename camera_t>
class SDLCamera : public Controllable_Camera<camera_t>
{
   public:
    SDLCamera()
    {
        this->keyboardmap = {
            //            SDL_SCANCODE_UP,
            //            SDL_SCANCODE_DOWN,
            //            SDL_SCANCODE_LEFT,
            //            SDL_SCANCODE_RIGHT,
            SDL_SCANCODE_W, SDL_SCANCODE_S, SDL_SCANCODE_A, SDL_SCANCODE_D, SDL_SCANCODE_LSHIFT, SDL_SCANCODE_SPACE};
        this->mousemap = {SDL_BUTTON_MIDDLE, SDL_BUTTON_LEFT};
    }
};

}  // namespace Saiga
