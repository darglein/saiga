/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/sdl/saiga_sdl.h"
#include "saiga/core/util/keyboard.h"
#include "saiga/core/util/mouse.h"

#include <vector>

#include "sdl_listener.h"

namespace Saiga
{
class SAIGA_CORE_API SDL_EventHandler
{
   private:
    friend class SDL_KeyListener;
    friend class SDL_MouseListener;
    friend class SDL_ResizeListener;
    friend class SDL_EventListener;


    static bool quit;
    static std::vector<SDL_KeyListener*> keyListener;
    static std::vector<SDL_MouseListener*> mouseListener;
    static std::vector<SDL_ResizeListener*> resizeListener;
    static std::vector<SDL_EventListener*> eventListener;

    static void reset();

   public:
    static void update();


    static void keyPressed(const SDL_Keysym& key);
    static void keyReleased(const SDL_Keysym& key);

    static void mouseMoved(int x, int y);
    static void mousePressed(int key, int x, int y);
    static void mouseReleased(int key, int x, int y);

    static void resizeWindow(Uint32 windowId, int width, int height);

    static bool shouldQuit() { return quit; }
};

}  // namespace Saiga
