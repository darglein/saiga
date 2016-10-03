#pragma once

#include <saiga/config.h>
#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include "saiga/sdl/sdl_listener.h"
#include <saiga/util/keyboard.h>
#include <saiga/util/mouse.h>

class SAIGA_GLOBAL SDL_EventHandler{
private:
    static bool quit;
    static std::vector<SDL_KeyListener*> keyListener;
    static std::vector<SDL_MouseListener*> mouseListener;
    static std::vector<SDL_EventListener*> eventListener;
public:
    static void addKeyListener(SDL_KeyListener* kl){keyListener.push_back(kl);}
    static void addMouseListener(SDL_MouseListener* ml){mouseListener.push_back(ml);}
    static void addEventListener(SDL_EventListener* ml){eventListener.push_back(ml);}
    static void update();


    static void keyPressed(const SDL_Keysym &key);
    static void keyReleased(const SDL_Keysym &key);

    static void mouseMoved(int x, int y);
    static void mousePressed(int key, int x, int y);
    static void mouseReleased(int key, int x, int y);
    static bool shouldQuit(){return quit;}
};

