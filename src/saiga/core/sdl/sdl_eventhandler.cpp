/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "sdl_eventhandler.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>

namespace Saiga
{
SDL_KeyListener::SDL_KeyListener()
{
    SDL_EventHandler::keyListener.push_back(this);
}

SDL_KeyListener::~SDL_KeyListener()
{
    auto& v = SDL_EventHandler::keyListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}

SDL_MouseListener::SDL_MouseListener()
{
    SDL_EventHandler::mouseListener.push_back(this);
}

SDL_MouseListener::~SDL_MouseListener()
{
    auto& v = SDL_EventHandler::mouseListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}


SDL_ResizeListener::SDL_ResizeListener()
{
    SDL_EventHandler::resizeListener.push_back(this);
}

SDL_ResizeListener::~SDL_ResizeListener()
{
    auto& v = SDL_EventHandler::resizeListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}


SDL_EventListener::SDL_EventListener()
{
    SDL_EventHandler::eventListener.push_back(this);
}

SDL_EventListener::~SDL_EventListener()
{
    auto& v = SDL_EventHandler::eventListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}


bool SDL_EventHandler::quit = false;
std::vector<SDL_KeyListener*> SDL_EventHandler::keyListener;
std::vector<SDL_MouseListener*> SDL_EventHandler::mouseListener;
std::vector<SDL_ResizeListener*> SDL_EventHandler::resizeListener;
std::vector<SDL_EventListener*> SDL_EventHandler::eventListener;

void SDL_EventHandler::update()
{
    // Handle events on queue
    SDL_Event e;
    while (SDL_PollEvent(&e) != 0)
    {
        // User requests quit
        if (e.type == SDL_QUIT)
        {
            quit = true;
        }
        if (e.type == SDL_KEYDOWN)
        {
            keyPressed(e.key.keysym);
        }
        if (e.type == SDL_KEYUP)
        {
            keyReleased(e.key.keysym);
        }
        /* If the mouse is moving */
        if (e.type == SDL_MOUSEMOTION)
        {
            mouseMoved(e.motion.x, e.motion.y);
        }
        if (e.type == SDL_MOUSEBUTTONDOWN)
        {
            int key = e.button.button;
            mousePressed(key, e.button.x, e.button.y);
        }
        if (e.type == SDL_MOUSEBUTTONUP)
        {
            int key = e.button.button;
            mouseReleased(key, e.button.x, e.button.y);
        }

        if (e.type == SDL_WINDOWEVENT)
        {
            SDL_WindowEvent we = e.window;
            if (we.event == SDL_WINDOWEVENT_RESIZED)
            {
                //                std::cout << "SDL Window (ID=" << we.windowID << ") resized to " << we.data1 << "x" <<
                //                we.data2 << std::endl;
                resizeWindow(we.windowID, we.data1, we.data2);
            }
        }

        for (SDL_EventListener* listener : eventListener)
        {
            listener->processEvent(e);
        }
    }
}

void SDL_EventHandler::keyPressed(const SDL_Keysym& key)
{
    keyboard.setKeyState(key.scancode, 1);
    for (SDL_KeyListener* listener : keyListener)
    {
        listener->keyPressed(key);
    }
}

void SDL_EventHandler::keyReleased(const SDL_Keysym& key)
{
    keyboard.setKeyState(key.scancode, 0);
    for (SDL_KeyListener* listener : keyListener)
    {
        listener->keyReleased(key);
    }
}

void SDL_EventHandler::mouseMoved(int x, int y)
{
    mouse.setPosition(ivec2(x, y));
    for (SDL_MouseListener* listener : mouseListener)
    {
        listener->mouseMoved(x, y);
    }
}

void SDL_EventHandler::mousePressed(int key, int x, int y)
{
    mouse.setKeyState(key, 1);
    for (SDL_MouseListener* listener : mouseListener)
    {
        listener->mousePressed(key, x, y);
    }
}

void SDL_EventHandler::mouseReleased(int key, int x, int y)
{
    mouse.setKeyState(key, 0);
    for (SDL_MouseListener* listener : mouseListener)
    {
        listener->mouseReleased(key, x, y);
    }
}

void SDL_EventHandler::resizeWindow(Uint32 windowId, int width, int height)
{
    for (SDL_ResizeListener* listener : resizeListener)
    {
        listener->resizeWindow(windowId, width, height);
    }
}

void SDL_EventHandler::reset()
{
    keyListener.clear();
    mouseListener.clear();
    resizeListener.clear();
    eventListener.clear();
    quit = false;
}

}  // namespace Saiga
