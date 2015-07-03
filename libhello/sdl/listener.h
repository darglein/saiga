#ifndef LISTENER_H
#define LISTENER_H

#include <libhello/config.h>
#include <SDL2/SDL.h>

class SAIGA_GLOBAL SDL_KeyListener{
public:
    virtual inline void keyPressed(const SDL_Event &e){keyPressed(e.key.keysym.sym);}
    virtual inline void keyReleased(const SDL_Event &e){keyReleased(e.key.keysym.sym);}

    virtual void keyPressed(int key) = 0;
    virtual void keyReleased(int key) = 0;
};

class SAIGA_GLOBAL SDL_MouseListener{
public:
    virtual void mouseMoved(int relx, int rely) = 0;
    virtual void mousePressed(int key, int x, int y) = 0;
    virtual void mouseReleased(int key, int x, int y) = 0;
};

#endif // LISTENER_H
