/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/opengl/OpenGLWindow.h"

#include "saiga/sdl/sdl_eventhandler.h"
#include <SDL2/SDL.h>

#undef main

namespace Saiga {


class SAIGA_GLOBAL SDLWindow : public OpenGLWindow, public SDL_ResizeListener{
public:

    SDL_Window* window = nullptr;
protected:
    SDL_GLContext gContext;

    virtual bool initWindow() override;
    virtual bool initInput() override;
    virtual bool shouldClose() override;
    virtual void checkEvents() override;
    virtual void swapBuffers() override;
    virtual void freeContext() override;
    virtual void loadGLFunctions() override;

    virtual bool resizeWindow(Uint32 windowId, int width, int height) override;
public:

    SDLWindow(WindowParameters windowParameters, OpenGLParameters openglParameter = OpenGLParameters());
    ~SDLWindow();

    virtual std::shared_ptr<ImGuiRenderer> createImGui() override;
};

}
