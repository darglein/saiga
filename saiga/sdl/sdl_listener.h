#ifndef LISTENER_H
#define LISTENER_H

#include <saiga/config.h>
#include <SDL2/SDL.h>

class SAIGA_GLOBAL SDL_KeyListener{
public:
    virtual ~SDL_KeyListener(){}

    virtual void keyPressed(SDL_Keysym key) = 0;
    virtual void keyReleased(SDL_Keysym key) = 0;
};

class SAIGA_GLOBAL SDL_MouseListener{
public:
    virtual ~SDL_MouseListener(){}
    virtual void mouseMoved(int x, int y) = 0;
    virtual void mousePressed(int key, int x, int y) = 0;
    virtual void mouseReleased(int key, int x, int y) = 0;
};

#endif // LISTENER_H
