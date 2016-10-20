#pragma once
#include "saiga/opengl/opengl.h"
#include <saiga/imgui/imgui.h>
#include <saiga/sdl/sdl_eventhandler.h>

// ImGui SDL2 binding with OpenGL3
// In this binding, ImTextureID is used to store an OpenGL 'GLuint' texture identifier. Read the FAQ about ImTextureID in imgui.cpp.

// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example of using this.
// If you use this binding you'll need to call 4 functions: ImGui_ImplXXXX_Init(), ImGui_ImplXXXX_NewFrame(), ImGui::Render() and ImGui_ImplXXXX_Shutdown().
// If you are new to ImGui, see examples/README.txt and documentation at the top of imgui.cpp.
// https://github.com/ocornut/imgui

struct SDL_Window;
typedef union SDL_Event SDL_Event;

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
