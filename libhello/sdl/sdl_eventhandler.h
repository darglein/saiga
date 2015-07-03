#pragma once

#include <libhello/config.h>
#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include "libhello/sdl/listener.h"

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


    void keyPressed(const SDL_Event &e);
    void keyReleased(const SDL_Event &e);

    void mouseMoved(int relx, int rely);
    void mousePressed(int key, int x, int y);
    void mouseReleased(int key, int x, int y);
    bool shouldQuit(){return quit;}
};

