/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/sdl/sdl_listener.h"
#include <saiga/util/keyboard.h>
#include <saiga/util/mouse.h>
#include <SDL2/SDL.h>
#include <vector>
 
namespace Saiga {

class SAIGA_GLOBAL SDL_EventHandler{
private:
    static bool quit;
    static std::vector<SDL_KeyListener*> keyListener;
    static std::vector<SDL_MouseListener*> mouseListener;
    static std::vector<SDL_ResizeListener*> resizeListener;
    static std::vector<SDL_EventListener*> eventListener;
public:
    static void addKeyListener(SDL_KeyListener* kl){keyListener.push_back(kl);}
    static void addMouseListener(SDL_MouseListener* ml){mouseListener.push_back(ml);}
    static void addResizeListener(SDL_ResizeListener* rl){resizeListener.push_back(rl);}
    static void addEventListener(SDL_EventListener* ml){eventListener.push_back(ml);}
    static void update();


    static void keyPressed(const SDL_Keysym &key);
    static void keyReleased(const SDL_Keysym &key);

    static void mouseMoved(int x, int y);
    static void mousePressed(int key, int x, int y);
    static void mouseReleased(int key, int x, int y);

    static void resizeWindow(Uint32 windowId, int width, int height);

    static bool shouldQuit(){return quit;}

    static void reset();
};

}
