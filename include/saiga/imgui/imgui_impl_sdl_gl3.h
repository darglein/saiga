//ImGui License:
//The MIT License (MIT)

//Copyright (c) 2014-2015 Omar Cornut and ImGui contributors

//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#pragma once

#include "saiga/opengl/opengl.h"
#include <saiga/imgui/imgui.h>


#ifdef SAIGA_USE_SDL
#include <saiga/sdl/sdl_eventhandler.h>
struct SDL_Window;
typedef union SDL_Event SDL_Event;

namespace Saiga {

// ImGui SDL2 binding with OpenGL3
// In this binding, ImTextureID is used to store an OpenGL 'GLuint' texture identifier. Read the FAQ about ImTextureID in imgui.cpp.

// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example of using this.
// If you use this binding you'll need to call 4 functions: ImGui_ImplXXXX_Init(), ImGui_ImplXXXX_NewFrame(), ImGui::Render() and ImGui_ImplXXXX_Shutdown().
// If you are new to ImGui, see examples/README.txt and documentation at the top of imgui.cpp.
// https://github.com/ocornut/imgui


class SAIGA_GLOBAL ImGui_SDL_Renderer : public SDL_EventListener{
protected:
    SDL_Window* window;
    static void ImGui_ImplSdlGL3_RenderDrawLists(ImDrawData *draw_data);
    void ImGui_ImplSdlGL3_CreateFontsTexture();
public:
    bool wantsCaptureMouse = false;

bool        init(SDL_Window* window, std::string font, float fontSize = 15.0f);
void        shutdown();
void        beginFrame();
void        endFrame();
void        checkWindowFocus();

// Use if you want to reset your rendering device without losing ImGui state.
void        ImGui_ImplSdlGL3_InvalidateDeviceObjects();
bool        ImGui_ImplSdlGL3_CreateDeviceObjects();
bool        processEvent(const SDL_Event& event);
};

}

#endif