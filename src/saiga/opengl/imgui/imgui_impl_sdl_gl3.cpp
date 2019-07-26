// ImGui License:
// The MIT License (MIT)

// Copyright (c) 2014-2015 Omar Cornut and ImGui contributors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// ImGui SDL2 binding with OpenGL3
// In this binding, ImTextureID is used to store an OpenGL 'GLuint' texture identifier. Read the FAQ about ImTextureID
// in imgui.cpp.

// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example of using this.
// If you use this binding you'll need to call 4 functions: ImGui_ImplXXXX_Init(), ImGui_ImplXXXX_NewFrame(),
// ImGui::Render() and ImGui_ImplXXXX_Shutdown(). If you are new to ImGui, see examples/README.txt and documentation at
// the top of imgui.cpp. https://github.com/ocornut/imgui

#include "imgui_impl_sdl_gl3.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/opengl.h"

#ifdef SAIGA_USE_SDL

#    include <SDL2/SDL.h>

#    include <SDL2/SDL_syswm.h>

namespace Saiga
{
static const char* ImGui_ImplSdlGL3_GetClipboardText(void* user_data)
{
    return SDL_GetClipboardText();
}

static void ImGui_ImplSdlGL3_SetClipboardText(void* user_data, const char* text)
{
    SDL_SetClipboardText(text);
}



ImGui_SDL_Renderer::ImGui_SDL_Renderer(SDL_Window* window, const ImGuiParameters& params) : ImGui_GL_Renderer(params)
{
    memset(g_MousePressed, 0, 3 * sizeof(bool));
    this->window = window;
    ImGuiIO& io  = ImGui::GetIO();
    io.KeyMap[ImGuiKey_Tab] =
        SDLK_TAB;  // Keyboard mapping. ImGui will use those indices to peek into the io.KeyDown[] array.
    io.KeyMap[ImGuiKey_LeftArrow]  = SDL_SCANCODE_LEFT;
    io.KeyMap[ImGuiKey_RightArrow] = SDL_SCANCODE_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow]    = SDL_SCANCODE_UP;
    io.KeyMap[ImGuiKey_DownArrow]  = SDL_SCANCODE_DOWN;
    io.KeyMap[ImGuiKey_PageUp]     = SDL_SCANCODE_PAGEUP;
    io.KeyMap[ImGuiKey_PageDown]   = SDL_SCANCODE_PAGEDOWN;
    io.KeyMap[ImGuiKey_Home]       = SDL_SCANCODE_HOME;
    io.KeyMap[ImGuiKey_End]        = SDL_SCANCODE_END;
    io.KeyMap[ImGuiKey_Delete]     = SDLK_DELETE;
    io.KeyMap[ImGuiKey_Backspace]  = SDLK_BACKSPACE;
    io.KeyMap[ImGuiKey_Enter]      = SDLK_RETURN;
    io.KeyMap[ImGuiKey_Escape]     = SDLK_ESCAPE;
    io.KeyMap[ImGuiKey_A]          = SDLK_a;
    io.KeyMap[ImGuiKey_C]          = SDLK_c;
    io.KeyMap[ImGuiKey_V]          = SDLK_v;
    io.KeyMap[ImGuiKey_X]          = SDLK_x;
    io.KeyMap[ImGuiKey_Y]          = SDLK_y;
    io.KeyMap[ImGuiKey_Z]          = SDLK_z;

    //    io.RenderDrawListsFn = ImGui_ImplSdlGL3_RenderDrawLists;   // Alternatively you can set this to NULL and call
    //    ImGui::GetDrawData() after ImGui::Render() to get the same ImDrawData pointer.
    io.RenderDrawListsFn  = 0;
    io.SetClipboardTextFn = ImGui_ImplSdlGL3_SetClipboardText;
    io.GetClipboardTextFn = ImGui_ImplSdlGL3_GetClipboardText;

#    ifdef _WIN32
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window, &wmInfo);
    io.ImeWindowHandle = wmInfo.info.win.window;
#    else
    (void)window;
#    endif



    std::cout << "Imgui Initialized!" << std::endl;
}



ImGui_SDL_Renderer::~ImGui_SDL_Renderer() {}



void ImGui_SDL_Renderer::beginFrame()
{
    ImGuiIO& io = ImGui::GetIO();

    // Setup display size (every frame to accommodate for window resizing)
    int w, h;
    int display_w, display_h;
    SDL_GetWindowSize(window, &w, &h);
    SDL_GL_GetDrawableSize(window, &display_w, &display_h);
    io.DisplaySize             = ImVec2((float)w, (float)h);
    io.DisplayFramebufferScale = ImVec2(w > 0 ? ((float)display_w / w) : 0, h > 0 ? ((float)display_h / h) : 0);

    // Setup time step
    Uint32 time         = SDL_GetTicks();
    double current_time = time / 1000.0;
    io.DeltaTime        = g_Time > 0.0 ? (float)(current_time - g_Time) : (float)(1.0f / 60.0f);
    if (io.DeltaTime <= 0) io.DeltaTime = (1.0f / 60.0f);
    g_Time = current_time;

    // Setup inputs
    // (we already got mouse wheel, keyboard keys & characters from SDL_PollEvent())
    int mx, my;
    Uint32 mouseMask = SDL_GetMouseState(&mx, &my);
    if (SDL_GetWindowFlags(window) & SDL_WINDOW_MOUSE_FOCUS)
        io.MousePos = ImVec2(
            (float)mx, (float)my);  // Mouse position, in pixels (set to -1,-1 if no mouse / on another screen, etc.)
    else
        io.MousePos = ImVec2(-1, -1);

    io.MouseDown[0] =
        g_MousePressed[0] || (mouseMask & SDL_BUTTON(SDL_BUTTON_LEFT)) !=
                                 0;  // If a mouse press event came, always pass it as "mouse held this frame", so we
                                     // don't miss click-release events that are shorter than 1 frame.
    io.MouseDown[1]   = g_MousePressed[1] || (mouseMask & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
    io.MouseDown[2]   = g_MousePressed[2] || (mouseMask & SDL_BUTTON(SDL_BUTTON_MIDDLE)) != 0;
    g_MousePressed[0] = g_MousePressed[1] = g_MousePressed[2] = false;

    io.MouseWheel = g_MouseWheel;
    g_MouseWheel  = 0.0f;

    // Hide OS mouse cursor if ImGui is drawing it
    SDL_ShowCursor(io.MouseDrawCursor ? 0 : 1);

    // Start the frame
    ImGui::NewFrame();
}



// This is the main rendering function that you have to implement and provide to ImGui (via setting up
// 'RenderDrawListsFn' in the ImGuiIO structure) If text or lines are blurry when integrating ImGui in your engine:
// - in your Render function, try translating your projection matrix by (0.5f,0.5f) or (0.375f,0.375f)

bool ImGui_SDL_Renderer::processEvent(const SDL_Event& event)
{
    ImGuiIO& io = ImGui::GetIO();
    switch (event.type)
    {
        case SDL_MOUSEWHEEL:
        {
            if (event.wheel.y > 0) g_MouseWheel = 1;
            if (event.wheel.y < 0) g_MouseWheel = -1;
            return true;
        }
        case SDL_MOUSEBUTTONDOWN:
        {
            if (event.button.button == SDL_BUTTON_LEFT) g_MousePressed[0] = true;
            if (event.button.button == SDL_BUTTON_RIGHT) g_MousePressed[1] = true;
            if (event.button.button == SDL_BUTTON_MIDDLE) g_MousePressed[2] = true;
            return true;
        }
        case SDL_TEXTINPUT:
        {
            ImGuiIO& io = ImGui::GetIO();
            io.AddInputCharactersUTF8(event.text.text);
            return true;
        }
        case SDL_KEYDOWN:
        case SDL_KEYUP:
        {
            int key          = event.key.keysym.sym & ~SDLK_SCANCODE_MASK;
            io.KeysDown[key] = (event.type == SDL_KEYDOWN);
            io.KeyShift      = ((SDL_GetModState() & KMOD_SHIFT) != 0);
            io.KeyCtrl       = ((SDL_GetModState() & KMOD_CTRL) != 0);
            io.KeyAlt        = ((SDL_GetModState() & KMOD_ALT) != 0);
            io.KeySuper      = ((SDL_GetModState() & KMOD_GUI) != 0);
            return true;
        }
    }
    return false;
}


}  // namespace Saiga
#endif
