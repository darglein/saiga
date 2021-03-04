/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/sdl/saiga_sdl.h"

namespace Saiga
{
class SAIGA_CORE_API Listener
{
   public:
    virtual ~Listener() {}

    bool listenerEnabled = true;
};

class SAIGA_CORE_API SDL_KeyListener : public Listener
{
   public:
    SDL_KeyListener();
    virtual ~SDL_KeyListener();

    virtual void keyPressed(SDL_Keysym key)  = 0;
    virtual void keyReleased(SDL_Keysym key) = 0;
};

class SAIGA_CORE_API SDL_MouseListener : public Listener
{
   public:
    SDL_MouseListener();
    virtual ~SDL_MouseListener();
    virtual void mouseMoved(int x, int y)             = 0;
    virtual void mousePressed(int key, int x, int y)  = 0;
    virtual void mouseReleased(int key, int x, int y) = 0;
};

class SAIGA_CORE_API SDL_ResizeListener : public Listener
{
   public:
    SDL_ResizeListener();
    virtual ~SDL_ResizeListener();
    virtual bool resizeWindow(Uint32 windowId, int width, int height) = 0;
};

class SAIGA_CORE_API SDL_EventListener : public Listener
{
   public:
    SDL_EventListener();
    virtual ~SDL_EventListener();
    virtual bool processEvent(const SDL_Event& event) = 0;
};

}  // namespace Saiga
