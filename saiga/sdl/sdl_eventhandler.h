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
    bool quit;
    std::vector<SDL_KeyListener*> keyListener;
    std::vector<SDL_MouseListener*> mouseListener;
public:
    SDL_EventHandler():quit(false){}
    void addKeyListener(SDL_KeyListener* kl){keyListener.push_back(kl);}
    void addMouseListener(SDL_MouseListener* ml){mouseListener.push_back(ml);}
    void update();


    void keyPressed(const SDL_Keysym &key);
    void keyReleased(const SDL_Keysym &key);

    void mouseMoved(int x, int y);
    void mousePressed(int key, int x, int y);
    void mouseReleased(int key, int x, int y);
    bool shouldQuit(){return quit;}
};

